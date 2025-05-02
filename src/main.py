import asyncio
import os
import json
import logging
import urllib.parse
import hashlib
import re

import httpx
from httpx_sse import aconnect_sse
from cachetools import TTLCache
from models import Models, Plan
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from icecream import ic
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_sse_client.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MCPClientSSE_Google")

load_dotenv()

# --- Exception Classes ---
class ConfigError(Exception): pass
class ConnectionError(Exception): pass
class ToolExecutionError(Exception): pass
class MaxToolIterationsError(Exception): pass

# --- Config Class ---
class Config:
    """Configuration class loading from .env for SSE client."""
    def __init__(self):
        load_dotenv()
        self.server_base_url = os.getenv("MCP_SERVER_BASE_URL")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.default_prompt = os.getenv("default_prompt") or """
        You are a helpful and diligent virtual assistant. For each user query, follow these steps:
        1. Analyze the query to identify all required actions (e.g., data retrieval, searches, storing results).
        2. Break down complex queries into sequential tool calls if multiple actions are needed.
        3. Use the appropriate tools from the available set: {functions}.
        4. ALWAYS USE the storage tool 'store_recon_results' to store the results from the tools 'portScanner', 'subDomainEnumerator', 'dnsEnumerator', 'webSearch' and 'webSearch4CVEs', WHEN ASKED to do so by the user.
        5. ALWAYS USE the storage tool 'process_and_ingest_documentation' when storing the documentation text for computer languages.
        6. ALWAYS USE the storage tool 'ingestText2DB' to store the results from the tool 'getWebPages', WHEN ASKED to do so by the user.
        7. For data retrieval or searches from the vector database, use the tool 'queryVectorDB'.
        8. Think step-by-step and provide intermediate results if tools are called.
        9. If no tools are needed, provide a direct answer.
        Ensure all actions are completed accurately and results are stored if requested.
        """
        if not self.server_base_url:
            raise ConfigError("MCP_SERVER_BASE_URL must be set in .env (e.g., http://localhost:8080)")
        if not self.google_api_key:
            raise ConfigError("GOOGLE_API_KEY must be set in .env")
        if not self.default_prompt:
            logger.warning("default_prompt not set in .env, using enhanced default.")
        
        if not self.server_base_url.startswith(("http://", "https://")):
            raise ConfigError("MCP_SERVER_BASE_URL should start with http:// or https://")
        self.server_base_url = self.server_base_url.rstrip('/')
        
        logger.info("SSE Client Configuration loaded successfully.")
        logger.info(f"Target MCP Server URL: {self.server_base_url}")

# --- MCP Client Class ---
class MCPClient:
    def __init__(self, max_tool_iterations=5, cache_size=10000, cache_ttl=36000):
        """Initialize the MCP SSE client with caching and conversation history."""
        try:
            logger.info("Initializing MCP SSE Client with Google Generative AI and Caching")
            self.config = Config()
            self.models = Models()
            self.llm = self.models.llm
            self.http_client = httpx.AsyncClient(timeout=None)
            self.tools_schema_for_binding: List[Dict[str, Any]] = []
            self.mcp_tools_map: Dict[str, Any] = {}
            self.system_prompt_template: str = ""
            self.default_prompt = self.config.default_prompt
            self.llm_with_tools = None
            self.max_tool_iterations = max_tool_iterations
            self.llm_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self.tool_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self.no_cache_tools = {"get_current_time", "fetch_realtime_data"}
            self.no_cache_query_patterns = ["current time", "latest news", "real-time"]
            self.conversation_history = []
            # Increased context size, configurable via .env
            self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", 100))
            self.max_tokens = int(os.getenv("MAX_TOKENS", 20000))
            self.last_tool_result = None
            logger.info(f"Initialized caches: LLM (size={cache_size}, ttl={cache_ttl}s), Tool (size={cache_size}, ttl={cache_ttl}s)")
            logger.info(f"Initialized conversation history: max_messages={self.max_history_messages}, max_tokens={self.max_tokens}")
        except ConfigError as e:
            logger.error(f"Configuration error during initialization: {e}")
            raise
        except ValueError as e:
            logger.error(f"Model initialization error: {e}")
            raise ConfigError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected initialization error: {e}", exc_info=True)
            raise

    def _estimate_tokens(self, messages: List[Any]) -> int:
        """Estimate token count for messages with improved accuracy."""
        total_tokens = 0
        for msg in messages:
            if not hasattr(msg, "content") or not isinstance(msg.content, str):
                continue
            # Split content into words and account for punctuation/special characters
            words = re.findall(r'\w+|[^\w\s]', msg.content, re.UNICODE)
            # Estimate: 1 token per word, 1 token per special character, adjust for short words
            total_tokens += len(words)
            # Add tokens for message overhead (e.g., role markers)
            total_tokens += 4  # Approximate overhead for role/content delimiters
        return total_tokens

    async def _summarize_history(self):
        """Summarize older conversation history to reduce token count while preserving critical details."""
        if len(self.conversation_history) > 50:  # Increased threshold
            try:
                summary_prompt = (
                    "Summarize the following conversation concisely, preserving key details such as:\n"
                    "- User queries and their intent.\n"
                    "- Tool calls, their arguments, and results (especially for SubDomainEnumerator, DnsEnumerator, etc.).\n"
                    "- Plan steps and their outcomes.\n"
                    "Omit redundant or repetitive information:\n" +
                    "\n".join(m.content[:200] for m in self.conversation_history[1:-10] if isinstance(m.content, str))
                )
                summary = await self.llm.ainvoke([
                    SystemMessage(content="You are a summarizer. Provide a concise summary of the conversation, retaining critical details about queries, tool calls, and plan steps."),
                    HumanMessage(content=summary_prompt)
                ])
                self.conversation_history = (
                    [self.conversation_history[0]] +
                    [HumanMessage(content=summary.content)] +
                    self.conversation_history[-10:]
                )
                logger.info(f"Summarized conversation history to {len(self.conversation_history)} messages")
            except Exception as e:
                logger.error(f"Failed to summarize conversation history: {e}", exc_info=True)
                self._truncate_history()

    def _truncate_history(self):
        """Truncate conversation history based on message count and token limit."""
        if len(self.conversation_history) > self.max_history_messages:
            system_messages = [m for m in self.conversation_history if isinstance(m, SystemMessage)]
            other_messages = [m for m in self.conversation_history if not isinstance(m, SystemMessage)]
            self.conversation_history = system_messages + other_messages[-(self.max_history_messages - len(system_messages)):]
            logger.debug(f"Truncated history to {len(self.conversation_history)} messages due to message limit")
        
        while self._estimate_tokens(self.conversation_history) > self.max_tokens and len(self.conversation_history) > 2:
            self.conversation_history = (
                [self.conversation_history[0]] +
                self.conversation_history[-(len(self.conversation_history) - 2):]
            )
            logger.debug(f"Truncated history to {len(self.conversation_history)} messages due to token limit")

    async def _prepare_system_prompt(self):
        """Prepare the system prompt template with tool descriptions."""
        prompts_url = f"{self.config.server_base_url}/mcp/prompts"
        tool_descriptions = "\n".join([
            f"- {name}: {info.get('description', 'No description')}"
            for name, info in self.mcp_tools_map.items()
        ])

        logger.info(f"Fetching prompt from server: {prompts_url}")
        try:
            async with aconnect_sse(self.http_client, "GET", prompts_url) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "prompt":
                        try:
                            data = json.loads(sse.data) if sse.data else {}
                            prompt = data['prompt']
                            self.system_prompt_template = prompt.replace('{functions}', tool_descriptions) if tool_descriptions else "No tools available."
                            logger.debug(f"System prompt template set:\n{self.system_prompt_template[:200]}...")
                            return
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON data for event '{sse.event}': {sse.data}")
                            data = {"raw_data": sse.data}
                    elif sse.event == "error":
                        error_message = data.get("error", f"Unknown error")
                        self.system_prompt_template = self.default_prompt.format(
                            functions=tool_descriptions if tool_descriptions else "No tools available."
                        )
                        logger.error(f"Error: {error_message}")
                        logger.debug(f"System prompt template set:\n{self.system_prompt_template[:200]}...")
                        return
        except httpx.RequestError as e:
            logger.error(f"HTTP error connecting to server at {prompts_url}: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to MCP server at {self.config.server_base_url}. Is it running?") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {prompts_url}: {e}")
            raise ConnectionError("Invalid JSON received from server for prompts.") from e
        except Exception as e:
            logger.error(f"Error fetching prompt from server: {e}", exc_info=True)
            raise ConnectionError(f"Failed to fetch prompt: {e}") from e

    def _validate_schema(self, params_schema: Dict[str, Any]) -> bool:
        """Validate a tool schema to ensure it meets minimum requirements."""
        if not isinstance(params_schema, dict):
            return False
        required_fields = ['type', 'properties']
        return all(field in params_schema for field in required_fields) and \
               isinstance(params_schema.get('properties'), dict)

    def _bind_tools_to_llm(self):
        """Bind the discovered MCP tools to the LangChain LLM with validation."""
        if not self.tools_schema_for_binding:
            logger.warning("No tool schemas found to bind to LLM.")
            self.llm_with_tools = self.llm
            return
        try:
            cleaned_schemas = []
            for schema in self.tools_schema_for_binding:
                if not isinstance(schema.get("parameters"), dict):
                    logger.warning(f"Skipping invalid schema for tool '{schema.get('name')}': parameters not a dict")
                    continue
                params = schema["parameters"].copy()
                params.pop("title", None)
                if not self._validate_schema(params):
                    logger.warning(f"Skipping invalid schema for tool '{schema.get('name')}': missing required fields")
                    continue
                cleaned_schemas.append({
                    "name": schema["name"],
                    "description": schema.get("description", f"Tool named {schema['name']}"),
                    "parameters": params
                })
            if not cleaned_schemas:
                logger.warning("No valid tool schemas after validation, using unbound LLM.")
                self.llm_with_tools = self.llm
                return
            self.llm_with_tools = self.llm.bind_tools(cleaned_schemas)
            logger.info(f"Successfully bound {len(cleaned_schemas)} tools to the LLM.")
        except Exception as e:
            logger.error(f"Failed to bind tools to LLM: {e}", exc_info=True)
            self.llm_with_tools = self.llm
            logger.info("Falling back to unbound LLM due to binding failure.")

    async def initialize_tools(self):
        """Fetch tool list and schemas from the running MCP SSE server."""
        tools_url = f"{self.config.server_base_url}/mcp/tools"
        logger.info(f"Fetching tool list from server: {tools_url}")
        try:
            async with aconnect_sse(self.http_client, "GET", tools_url) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "tools_list":
                        logger.debug(f"Received tools_list event: {sse.data}")
                        data = json.loads(sse.data)
                        if "available_tools" not in data or not isinstance(data["available_tools"], list):
                            raise ConnectionError("Received invalid format for tools_list event from server.")
                        server_tools = data["available_tools"]
                        logger.info(f"Received {len(server_tools)} tool definitions from server.")
                        self.mcp_tools_map = {tool['name']: tool for tool in server_tools}
                        self.tools_schema_for_binding = []
                        for tool in server_tools:
                            if not tool.get('name') or not isinstance(tool.get('parameters'), dict):
                                logger.warning(f"Skipping tool due to missing name or invalid parameter schema: {tool.get('name')}")
                                continue
                            if not self._validate_schema(tool['parameters']):
                                logger.warning(f"Skipping tool '{tool.get('name')}' due to invalid schema")
                                continue
                            self.tools_schema_for_binding.append({
                                "name": tool['name'],
                                "description": tool.get('description', f"Tool named {tool['name']}"),
                                "parameters": tool['parameters']
                            })
                        logger.info(f"Processed tools: {list(self.mcp_tools_map.keys())}")
                        print(f"\nConnected to server. Tools: {list(self.mcp_tools_map.keys())}")
                        
                        self._bind_tools_to_llm()
                        self.models.initializeTools(self.tools_schema_for_binding)
                        await self._prepare_system_prompt()
                        return
                    elif sse.event == "error":
                        logger.error(f"Server sent error while fetching tools: {sse.data}")
                        raise ConnectionError(f"Server error fetching tools: {sse.data}")
        except httpx.RequestError as e:
            logger.error(f"HTTP error connecting to server at {tools_url}: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to MCP server at {self.config.server_base_url}. Is it running?") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {tools_url}: {e}")
            raise ConnectionError("Invalid JSON received from server for tools list.") from e
        except Exception as e:
            logger.error(f"Error initializing tools from server: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize tools: {e}") from e

    def _should_bypass_cache(self, query: str, tool_name: Optional[str] = None) -> bool:
        """Determine if caching should be bypassed for a query or tool."""
        if tool_name and tool_name in self.no_cache_tools:
            logger.debug(f"Bypassing cache for tool: {tool_name}")
            return True
        if any(pattern.lower() in query.lower() for pattern in self.no_cache_query_patterns):
            logger.debug(f"Bypassing cache for query pattern match: {query[:100]}...")
            return True
        return False

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_str = json.dumps([args, kwargs], sort_keys=True)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    async def _execute_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def execute_with_retry():
            encoded_args = urllib.parse.quote(json.dumps(tool_args))
            tool_url = f"{self.config.server_base_url}/mcp/tools/{tool_name}?arguments={encoded_args}"
            logger.debug(f"Tool execution URL: {tool_url}")
            
            final_result = None
            error_message = None
            status_messages = []
            
            async with aconnect_sse(self.http_client, "GET", tool_url) as event_source:
                async for sse in event_source.aiter_sse():
                    logger.debug(f"SSE received: event='{sse.event}', data='{sse.data[:200]}...'")
                    try:
                        data = json.loads(sse.data) if sse.data else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON data for event '{sse.event}': {sse.data}")
                        data = {"raw_data": sse.data}
                    
                    if sse.event == "error":
                        error_message = data.get("error", f"Unknown error from tool '{tool_name}'")
                        logger.error(f"Tool '{tool_name}' reported error: {error_message}")
                        break
                    
                    elif sse.event == "status":
                        status_messages.append(data.get("status", data.get("message", "Status update")))
                    
                    elif sse.event == "final_result":
                        final_result = data
                        logger.info(f"Received final_result event for {tool_name}")
                        break
            
            if error_message:
                logger.error(f"Tool execution failed for '{tool_name}'. Error: {error_message}")
                return {"error": error_message, "status_messages_received": status_messages}
            elif final_result is not None:
                # Validate the final_result
                if not final_result or (isinstance(final_result, dict) and not final_result):
                    logger.warning(f"Tool '{tool_name}' returned empty result. Returning default response.")
                    return {"status": "completed_with_empty_result", "status_updates": status_messages}
                logger.info(f"Tool '{tool_name}' execution successful.")
                return final_result
            else:
                logger.warning(f"Tool '{tool_name}' stream ended without explicit final result or error. Statuses: {status_messages}")
                return {"status": "completed_without_explicit_result", "status_updates": status_messages}
        
        if self._should_bypass_cache("", tool_name):
            logger.info(f"Cache bypassed for tool '{tool_name}'")
            try:
                return await execute_with_retry()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.error("Rate limit exceeded, retry attempts exhausted")
                    return {"error": "Rate limit exceeded"}
                raise
            except Exception as e:
                logger.error(f"Error processing SSE stream for tool '{tool_name}': {e}", exc_info=True)
                return {"error": f"Error processing tool '{tool_name}' response: {str(e)}"}
        
        cache_key = self._generate_cache_key(tool_name, tool_args)
        cached_result = self.tool_cache.get(cache_key, None)
        if cached_result is not None:
            logger.debug(f"Tool cache hit for {tool_name} with key {cache_key}")
            return cached_result
        
        logger.debug(f"Tool cache miss for {tool_name} with key {cache_key}, calling remote SSE tool")
        logger.info(f"Requesting execution of MCP tool '{tool_name}' via SSE (cache miss)")
        if tool_name not in self.mcp_tools_map:
            logger.error(f"Attempted to call unknown tool: {tool_name}")
            return {"error": f"Tool '{tool_name}' is not available."}
        
        try:
            final_result = await execute_with_retry()
            self.tool_cache[cache_key] = final_result
            return final_result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error calling tool '{tool_name}': {e.response.status_code} - {e.response.text[:200]}")
            result = {"error": f"Server error {e.response.status_code} calling tool '{tool_name}'."}
            self.tool_cache[cache_key] = result
            return result
        except httpx.ReadTimeout:
            logger.error(f"Read timeout waiting for SSE events from tool '{tool_name}'")
            result = {"error": f"Timeout waiting for response from tool '{tool_name}'"}
            self.tool_cache[cache_key] = result
            return result
        except httpx.RequestError as e:
            logger.error(f"HTTP error calling tool '{tool_name}': {e}", exc_info=True)
            result = {"error": f"Network error calling tool '{tool_name}': {str(e)}"}
            self.tool_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error processing SSE stream for tool '{tool_name}': {e}", exc_info=True)
            result = {"error": f"Error processing tool '{tool_name}' response: {str(e)}"}
            self.tool_cache[cache_key] = result
            return result

    async def _invoke_llm(self, messages: List[Any]) -> AIMessage:
        """Invoke the LLM with manual async caching and retries."""
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def invoke_with_retry():
            response = await self.llm_with_tools.ainvoke(messages)
            logger.debug(f"Raw LLM response: content='{response.content}', tool_calls={response.tool_calls}, metadata={response.response_metadata}")
            return response
        
        latest_query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        system_prompt = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        cache_key = self._generate_cache_key(system_prompt, latest_query)
        cached_value = self.llm_cache.get(cache_key, None)
        if cached_value is not None:
            logger.debug("LLM cache hit for key=%s", cache_key)
            return cached_value
        logger.debug("LLM cache miss for key=%s, invoking model", cache_key)
        logger.info("Invoking LLM (cache miss)")
        try:
            result = await invoke_with_retry()
            self.llm_cache[cache_key] = result
            return result
        except Exception as e:
            if "rate limit" in str(e).lower():
                logger.error("LLM rate limit exceeded, retry attempts exhausted")
                raise
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise

    def _format_tool_result(self, tool_result: Any, tool_name: str) -> str:
        """Format tool result for readable output (robust to list/dict/str results)."""
        if isinstance(tool_result, dict) and 'error' in tool_result:
            return f"Error from {tool_name}: {tool_result['error']}"
        elif isinstance(tool_result, dict) and len(tool_result) > 0:
            try:
                readable_response = json.dumps(tool_result, indent=2)
            except Exception:
                readable_response = str(tool_result)
        elif isinstance(tool_result, list) and len(tool_result) > 0:
            readable_response = "\n".join(str(item) for item in tool_result[:50])
            if len(tool_result) > 50:
                readable_response += f"\n\n...and {len(tool_result) - 50} more (showing first 50 only)"
        else:
            readable_response = str(tool_result) or "Tool executed successfully, no output provided."
        return readable_response

    async def _execute_plan(self, ai_plan: Plan, executed_tool_calls: set, final_results: list):
        """Recursively executes steps defined in a Plan object."""
        if not isinstance(ai_plan, Plan) or not hasattr(ai_plan, 'plan'):
            logger.error(f"Invalid plan structure received: {ai_plan}")
            return "(Error: Invalid plan structure received from LLM)"

        try:
            for step in ai_plan.plan:
                if step.prompt:
                    current_prompt = ic(step.prompt)
                    # Append last tool result to prompt if available
                    if self.last_tool_result and isinstance(self.last_tool_result, dict) and 'result' in self.last_tool_result:
                        tool_result_summary = self._format_tool_result(self.last_tool_result, 'last_tool')
                        current_prompt += f"\nPrevious step result:\n{tool_result_summary}"
                    print(f"\nExecuting Step: {current_prompt}")
                    logger.info(f"Executing plan step with prompt: {current_prompt[:100]}...")
                    self.conversation_history.append(HumanMessage(content=current_prompt))
                    await self._summarize_history()
                    self._truncate_history()

                    valid_history = []
                    original_length = len(self.conversation_history)
                    for msg in self.conversation_history:
                        keep_message = False
                        # Keep System/Human messages if they have non-empty content
                        if isinstance(msg, (SystemMessage, HumanMessage)):
                            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                                keep_message = True
                            else:
                                logger.debug(f"Filtering message (System/Human) due to empty content: type={type(msg)}")
                        # Keep AIMessage if it has non-empty content OR tool_calls
                        elif isinstance(msg, AIMessage):
                            has_content = hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip()
                            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls and len(msg.tool_calls) > 0
                            if has_content or has_tool_calls:
                                keep_message = True
                            else:
                                logger.debug(f"Filtering AIMessage lacking content and tool_calls: {msg}")
                        # Keep ToolMessage only if content is non-empty and valid
                        elif isinstance(msg, ToolMessage):
                            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                                keep_message = True
                            else:
                                logger.debug(f"Filtering ToolMessage with empty or invalid content: {msg}")
                        else:
                            # Keep any other unexpected message types, but log a warning
                            logger.warning(f"Encountered unexpected message type in history: {type(msg)}. Keeping it.")
                            keep_message = True

                        if keep_message:
                            valid_history.append(msg)

                    filtered_count = original_length - len(valid_history)
                    if filtered_count > 0:
                        logger.info(f"Filtered {filtered_count} messages based on type/content/tool_calls.")

                    # Ensure history isn't completely empty after filtering
                    if not valid_history:
                        logger.error("Conversation history became empty after filtering! Cannot invoke LLM.")
                        final_results.append("(Critical Error: Conversation history became empty)")
                        return "(Critical Error: Conversation history became empty)"

                    ai_response = ic(await self._invoke_llm(valid_history))
                    self.conversation_history.append(ai_response)
                    self._truncate_history()

                    tool_calls = getattr(ai_response, "tool_calls", [])
                    ai_content = ai_response.content if isinstance(ai_response.content, str) else ""

                    if tool_calls:
                        logger.info(f"Processing {len(tool_calls)} tool call(s): {[tc.get('name') for tc in tool_calls]}")
                        results_added_this_step = False
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id")

                            if not tool_name or not isinstance(tool_args, dict) or not tool_call_id:
                                logger.error(f"Malformed tool call detected: {tool_call}")
                                tool_message = ToolMessage(
                                    content=f"Error: Malformed tool call: {tool_call}",
                                    tool_call_id=tool_call_id or f"error_step_{step.prompt[:20]}"
                                )
                                self.conversation_history.append(tool_message)
                                self._truncate_history()
                                final_results.append(f"Error executing step '{current_prompt}': Malformed tool call.")
                                results_added_this_step = True
                                continue

                            tool_call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
                            if tool_call_key in executed_tool_calls:
                                logger.warning(f"Skipping redundant tool call: {tool_name} with args {tool_args}")
                                continue
                            executed_tool_calls.add(tool_call_key)

                            tool_result = await self._execute_mcp_tool(tool_name, tool_args)
                            self.last_tool_result = tool_result
                            # Sanitize tool result content
                            try:
                                tool_result_content = json.dumps(tool_result)
                                if not tool_result_content.strip():
                                    logger.warning(f"Tool '{tool_name}' returned empty JSON content. Using default message.")
                                    tool_result_content = json.dumps({"status": "tool_executed", "message": "No content returned"})
                            except (TypeError, ValueError) as e:
                                logger.error(f"Failed to serialize tool result for '{tool_name}': {e}")
                                tool_result_content = json.dumps({"error": f"Failed to serialize tool result: {str(e)}"})

                            # Smart truncation for large results
                            max_result_len = 50000
                            if len(tool_result_content) > max_result_len:
                                logger.warning(f"Tool result for {tool_name} is too long ({len(tool_result_content)} chars), applying smart truncation.")
                                if isinstance(tool_result, dict) and 'result' in tool_result and 'subdomains' in tool_result['result']:
                                    subdomains = tool_result['result']['subdomains']
                                    truncated_subdomains = subdomains[:100]
                                    tool_result['result']['subdomains'] = truncated_subdomains
                                    tool_result['truncated'] = f"Subdomains truncated to {len(truncated_subdomains)} from {len(subdomains)}"
                                    tool_result_content = json.dumps(tool_result)
                                if len(tool_result_content) > max_result_len:
                                    tool_result_content = tool_result_content[:max_result_len] + "...(truncated)"

                            tool_message = ToolMessage(
                                content=tool_result_content,
                                tool_call_id=tool_call_id
                            )

                            self.conversation_history.append(tool_message)
                            self._truncate_history()
                            results_added_this_step = True
                            logger.debug(f"Appended ToolMessage for call_id {tool_call_id}")

                            if isinstance(tool_result, dict) and 'error' in tool_result:
                                error_msg = f"Tool '{tool_name}' failed: {tool_result['error']}"
                                logger.error(error_msg)
                                final_results.append(f"Error executing step '{current_prompt}': {error_msg}")

                        if not results_added_this_step and tool_calls:
                            logger.warning(f"All tool calls for step '{current_prompt}' were skipped.")
                            final_results.append(f"Step '{current_prompt}' completed (tool calls skipped).")

                    else:
                        logger.info(f"No tool calls for step '{current_prompt}'. Using LLM content.")
                        print(f"Assistant (Step Result): {ai_content}")
                        final_results.append(ai_content or f"Step '{current_prompt}' completed.")

                else:
                    logger.error(f"Invalid step structure encountered (neither prompt nor plan): {step}")
                    final_results.append(f"(Error: Invalid step structure in plan)")

            last_ai_message = next((msg for msg in reversed(self.conversation_history) if isinstance(msg, AIMessage) and not msg.tool_calls), None)
            if last_ai_message and isinstance(last_ai_message.content, str):
                return last_ai_message.content
            elif self.last_tool_result:
                return f"Plan completed. Last action result:\n{self._format_tool_result(self.last_tool_result, 'last_tool')}"
            else:
                return "(Plan executed successfully)"

        except Exception as e:
            logger.error(f"Error during plan execution: {e}", exc_info=True)
            if "rate limit" in str(e).lower():
                return "(Error: API Rate Limit Exceeded during plan execution.)"
            return f"Error executing plan: {str(e)}"

    async def process_query(self, query: str) -> str:
        """Process a user query with tool calls and conversation history."""
        if not self.llm_with_tools:
            logger.error("LLM with tools is not initialized.")
            return "Error: LLM tools binding failed."
        if not self.models.structured_llm:
            logger.error("Structured LLM with tools is not initialized properly.")
            return "Error: LLM setup failed. Cannot generate plan."
        if not query.strip():
            return "Query cannot be empty"
        
        logger.info(f"Processing query: {query[:100]}...")
        if not self.conversation_history:
            if not self.system_prompt_template:
                await self._prepare_system_prompt()
                if not self.system_prompt_template:
                    logger.error("System prompt template could not be loaded.")
                    return "Error: Failed to load system prompt."
            self.conversation_history.append(SystemMessage(content=self.system_prompt_template))

        self.conversation_history.append(HumanMessage(content=query))
        await self._summarize_history()
        self._truncate_history()
        
        valid_history = [msg for msg in self.conversation_history if hasattr(msg, "content") and str(msg.content).strip()]
        if len(valid_history) != len(self.conversation_history):
            logger.info(f"Filtered {len(self.conversation_history) - len(valid_history)} messages with empty content")
        
        try:
            executed_tool_calls = set()
            final_results = []
            ai_plan = ic(await self.models.invoke_structured_plan(valid_history))
            plan_results = await self._execute_plan(ai_plan, executed_tool_calls, final_results)
            final_results.append(plan_results)
            final_results.reverse()
            return final_results
        except RuntimeError as e:
            logger.error(f"Error generating or validating structured plan: {e}", exc_info=True)
            return f"Error processing query: {e}"
        except MaxToolIterationsError as e:
            logger.error(f"Max tool iterations reached: {e}")
            return f"\nError: {e}"
        except Exception as e:
            logger.error(f"Error during LLM invocation or plan processing: {e}", exc_info=True)
            if "rate limit" in str(e).lower():
                return "(Error: API Rate Limit Exceeded. Please wait and try again.)"
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop with conversation context."""
        if not self.llm_with_tools:
            logger.error("Cannot start chat: LLM or tools not initialized.")
            print("\nError: Could not initialize tools from server. Exiting.")
            return
        
        logger.info("Starting interactive chat loop (SSE Client - Google Gemini Pro)")
        print("\nMCP SSE Client (Google Gemini Pro) Started!")
        print(f"Connected to: {self.config.server_base_url}")
        print(f"Using Tools: {list(self.mcp_tools_map.keys())}")
        print("Conversation context is maintained across queries.")
        print("Type your queries, 'quit' to exit, or 'clear' to reset conversation history.")
        
        while True:
            try:
                query = await asyncio.to_thread(input, "\nQuery: ")
                query = query.strip()
                if query.lower() in ['quit', 'exit', 'bye']:
                    logger.info("User requested to quit chat loop")
                    break
                if query.lower() == 'clear':
                    self.conversation_history = []
                    self.last_tool_result = None
                    logger.info("Conversation history and last tool result cleared by user")
                    print("Conversation history cleared.")
                    continue
                if not query:
                    continue
                
                logger.debug(f"User query: {query[:100]}...")
                response = await self.process_query(query)
                print("\nAssistant:")
                print(response)
            
            except KeyboardInterrupt:
                logger.warning("Chat loop interrupted by user (Ctrl+C)")
                print("\nExiting chat loop...")
                break
            except MaxToolIterationsError as e:
                logger.error(f"Max tool iterations reached: {e}")
                print(f"\nError: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in chat loop: {e}", exc_info=True)
                print(f"\nAn unexpected error occurred: {str(e)}")
                print("Type 'quit' to exit, 'clear' to reset history, or try another query.")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up MCP SSE client resources")
        try:
            await self.http_client.aclose()
            logger.info("HTTP client closed.")
            self.llm_cache.clear()
            self.tool_cache.clear()
            self.conversation_history = []
            self.last_tool_result = None
            logger.info("Caches, conversation history, and last tool result cleared.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

# --- Main Function ---
async def main():
    """Main entry point for the SSE client application."""
    client = None
    try:
        logger.info("--- Starting MCP SSE Client Application (Google Gemini) ---")
        client = MCPClient()
        await client.initialize_tools()
        await client.chat_loop()
    except ConfigError as e:
        logger.critical(f"Configuration error: {e}")
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file (MCP_SERVER_BASE_URL, GOOGLE_API_KEY, default_prompt).")
    except ConnectionError as e:
        logger.critical(f"Connection or Initialization error: {e}")
        print(f"\nConnection/Initialization Error: {e}")
        server_url = client.config.server_base_url if client and hasattr(client, 'config') else 'N/A'
        print(f"Could not connect to the MCP server at {server_url} or retrieve/initialize tools.")
        print("Ensure the MCP FastAPI server is running and accessible, and check server logs.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if client:
            await client.cleanup()
        logger.info("--- MCP SSE Client Application Shutdown Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        logger.warning("Application terminated by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\nFatal error during application startup or shutdown: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
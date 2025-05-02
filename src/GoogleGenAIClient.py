# --- START OF FILE main.py ---

import asyncio
import os
import json
import logging
from typing import Optional, List, Dict, Any
import urllib.parse
import hashlib

import httpx
# **** Import aconnect_sse ****
from httpx_sse import aconnect_sse, SSEError
from cachetools import TTLCache
# Import models AFTER httpx_sse in case models imports httpx implicitly later
from GoogleGenAIModels import Models, Plan, ToolStep # Import Plan and ToolStep
from dotenv import load_dotenv
# Import LangChain message types AND Google types where needed
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from google.generativeai import types as google_types # Alias for clarity

from tenacity import retry, stop_after_attempt, wait_exponential
from icecream import ic

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
        # Corrected prompt key name to match .env standard practice
        self.default_prompt_template = os.getenv("DEFAULT_PROMPT_TEMPLATE") or """
        You are a helpful and diligent virtual assistant. For each user query, follow these steps:
        1. Analyze the query to identify all required actions (e.g., data retrieval, searches, storing results).
        2. Break down complex queries into sequential tool calls if multiple actions are needed.
        3. Use the appropriate tools from the available set: {functions}.
        4. ALWAYS USE the storage tool 'store_recon_results' to store the results from the tools 'portScanner' , 'subDomainEnumerator' , 'dnsEnumerator', 'webSearch' and 'webSearch4CVEs', WHEN ASKED to do so by the user.
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
        if not self.default_prompt_template: # Check the corrected name
            logger.warning("DEFAULT_PROMPT_TEMPLATE not set in .env, using enhanced default.")

        if not self.server_base_url.startswith(("http://", "https://")):
            raise ConfigError("MCP_SERVER_BASE_URL should start with http:// or https://")
        self.server_base_url = self.server_base_url.rstrip('/')

        logger.info("SSE Client Configuration loaded successfully.")
        logger.info(f"Target MCP Server URL: {self.server_base_url}")

# --- MCP Client Class ---
class MCPClient:
    def __init__(self, max_tool_iterations=5, cache_size=1000, cache_ttl=3600):
        """Initialize the MCP SSE client with caching and conversation history."""
        try:
            logger.info("Initializing MCP SSE Client with Google Generative AI and Caching")
            self.config = Config()
            # Initialize Models directly
            self.models = Models() # LLM instance is inside Models
            self.http_client = httpx.AsyncClient(timeout=30.0)
            self.tools_schema_for_binding: List[Dict[str, Any]] = []
            self.mcp_tools_map: Dict[str, Any] = {}
            self.system_prompt_template: str = "" # This will be formatted later
            self.default_prompt = self.config.default_prompt_template # Use the loaded template
            self.max_tool_iterations = max_tool_iterations
            self.llm_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self.tool_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self.no_cache_tools = {"get_current_time", "fetch_realtime_data"}
            self.no_cache_query_patterns = ["current time", "latest news", "real-time"]
            self.conversation_history = [] # List of LangChain messages
            self.max_history_messages = 50
            self.max_tokens = 7000 # Approximate token limit
            self.last_tool_result = None # Store the raw result from the tool execution
            logger.info(f"Initialized caches: LLM (size={cache_size}, ttl={cache_ttl}s), Tool (size={cache_size}, ttl={cache_ttl}s)")
            logger.info(f"Initialized conversation history: max_messages={self.max_history_messages}, max_tokens={self.max_tokens}")
        except ConfigError as e:
            logger.error(f"Configuration error during initialization: {e}")
            raise
        except ValueError as e:
            logger.error(f"Model initialization error: {e}")
            # Wrap in ConfigError or a more specific InitializationError if desired
            raise ConfigError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected initialization error: {e}", exc_info=True)
            raise

    def _estimate_tokens(self, messages: List[Any]) -> int:
        """Estimate token count for LangChain messages using a simple approximation."""
        # This is very rough for Gemini. Use the count_tokens method if accuracy is critical.
        text_content = ""
        num_messages = len(messages)
        for m in messages:
            if isinstance(m, (SystemMessage, HumanMessage, AIMessage)) and isinstance(m.content, str):
                text_content += m.content + " "
            elif isinstance(m, ToolMessage) and isinstance(m.content, str):
                text_content += m.content + " " # Count tool results too
            elif isinstance(m, AIMessage) and m.tool_calls:
                 # Add rough estimate for tool call structure
                 text_content += json.dumps([tc['name'] for tc in m.tool_calls]) + " "

        # Rough estimate: 4 chars per token (adjust based on observation)
        return len(text_content) // 4 + num_messages * 5 # Add overhead per message

    # --- History Management (Keep _summarize_history and _truncate_history as they are) ---
    async def _summarize_history(self):
        """Summarize older conversation history to reduce token count."""
        if len(self.conversation_history) > 20: # Only summarize if history is long
            # Check token count before summarizing
            current_tokens = self._estimate_tokens(self.conversation_history)
            if current_tokens < self.max_tokens * 0.8: # Don't summarize if well below limit
                logger.debug("Skipping summarization, token count below threshold.")
                return

            try:
                # Select messages to summarize (e.g., messages 1 to -10)
                messages_to_summarize = self.conversation_history[1:-10]
                if not messages_to_summarize: return # Nothing to summarize

                logger.info(f"Summarizing {len(messages_to_summarize)} messages...")

                # Prepare prompt for summarization (using invoke_prompt logic)
                summary_prompt_text = "Summarize the key information and outcomes from the following conversation excerpt concisely:\n"
                summary_formatted_history = []
                for msg in messages_to_summarize:
                     # Simplified formatting for summarizer
                     if isinstance(msg, HumanMessage):
                         summary_formatted_history.append(f"User: {msg.content}")
                     elif isinstance(msg, AIMessage):
                         summary_formatted_history.append(f"Assistant: {msg.content}")
                     elif isinstance(msg, ToolMessage):
                          summary_formatted_history.append(f"Tool Result ({msg.tool_call_id}): {msg.content[:200]}...") # Truncate tool results

                summary_prompt_text += "\n".join(summary_formatted_history)

                # Use a basic LLM call for summarization (formatted correctly)
                summarization_messages = [
                    # System message for summarizer role
                    {'role': 'user', 'parts': ["You are a summarization assistant. Provide a concise summary of the given conversation excerpt."]},
                    {'role': 'user', 'parts': [summary_prompt_text]}
                ]

                # Use the base model without tools for summarization
                summary_response = await self.models.llm.generate_content_async(
                    contents=summarization_messages
                )
                summary_text = summary_response.text

                # Replace summarized messages with the summary
                self.conversation_history = (
                    [self.conversation_history[0]] + # Keep original system message
                    [AIMessage(content=f"[Summarized History]:\n{summary_text}")] + # Represent summary as AI message
                    self.conversation_history[-10:] # Keep recent messages
                )
                logger.info(f"Summarized conversation history to {len(self.conversation_history)} messages. New token estimate: {self._estimate_tokens(self.conversation_history)}")

            except Exception as e:
                logger.error(f"Failed to summarize conversation history: {e}", exc_info=True)
                # Fallback to simple truncation if summarization fails
                self._truncate_history(force_truncate=True) # Force truncation

    def _truncate_history(self, force_truncate=False):
        """Truncate conversation history based on message count and token limit."""
        truncated = False
        # Truncate by message count first
        if len(self.conversation_history) > self.max_history_messages:
            system_messages = [m for m in self.conversation_history if isinstance(m, SystemMessage)]
            other_messages = [m for m in self.conversation_history if not isinstance(m, SystemMessage)]
            keep_count = self.max_history_messages - len(system_messages)
            if keep_count < 1: keep_count = 1 # Keep at least one message besides system
            self.conversation_history = system_messages + other_messages[-keep_count:]
            logger.debug(f"Truncated history to {len(self.conversation_history)} messages due to message limit")
            truncated = True

        # Truncate by token count
        current_tokens = self._estimate_tokens(self.conversation_history)
        while current_tokens > self.max_tokens and len(self.conversation_history) > 2: # Keep at least System + 1 message
            # Remove the oldest non-system message (usually the second element)
            system_messages = [m for m in self.conversation_history if isinstance(m, SystemMessage)]
            other_messages = [m for m in self.conversation_history if not isinstance(m, SystemMessage)]
            if len(other_messages) > 1:
                other_messages.pop(0) # Remove the oldest interaction message
                self.conversation_history = system_messages + other_messages
                new_tokens = self._estimate_tokens(self.conversation_history)
                logger.debug(f"Truncated history due to token limit. New count: {len(self.conversation_history)}, Tokens: {new_tokens}")
                current_tokens = new_tokens
                truncated = True
            else:
                # Should not happen if check is len > 2, but prevents infinite loop
                logger.warning("Cannot truncate further, only system message and one other message remain.")
                break

        if truncated or force_truncate:
             logger.info(f"History adjusted. Current size: {len(self.conversation_history)} messages, Estimated tokens: {self._estimate_tokens(self.conversation_history)}")

    # --- _prepare_system_prompt remains mostly the same, uses stream/aiter_lines which is ok for prompts ---
    async def _prepare_system_prompt(self):
        """Prepare the system prompt template with tool descriptions."""
        # Ensure mcp_tools_map is populated first
        if not self.mcp_tools_map:
            logger.error("Cannot prepare system prompt: Tool map is empty. Initialize tools first.")
            # Use a generic default if tools aren't loaded
            self.system_prompt_template = self.default_prompt.format(functions="No tools available.")
            return

        prompts_url = f"{self.config.server_base_url}/mcp/prompts"
        tool_descriptions = "\n".join([
            f"- {name}: {info.get('description', 'No description')}"
            for name, info in self.mcp_tools_map.items()
        ])

        logger.info(f"Fetching prompt template from server: {prompts_url}")
        base_prompt_template = self.default_prompt # Start with the default from config

        try:
            # Try fetching from server, but have the default as fallback
            # Using httpx.stream is acceptable here as it handles line-by-line processing
            async with self.http_client.stream("GET", prompts_url) as response:
                response.raise_for_status() # Check for HTTP errors
                async for line in response.aiter_lines():
                    if line.startswith("event: prompt"):
                        # Next line should be data
                        continue
                    if line.startswith("data:"):
                        try:
                            data_json = line[len("data:"):].strip()
                            data = json.loads(data_json) if data_json else {}
                            prompt_from_server = data.get('prompt')
                            if prompt_from_server and isinstance(prompt_from_server, str):
                                base_prompt_template = prompt_from_server
                                logger.info("Successfully fetched prompt template from server.")
                                break # Got the prompt, exit loop
                            else:
                                logger.warning(f"Received invalid prompt data from server: {data_json}")
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON prompt data from server: {line}")
                    elif line.startswith("event: error"):
                         logger.error(f"Server sent error event while fetching prompt: {line}")
                         # Keep using the default template
                         break

            # Format the chosen template (server or default) with tool descriptions
            self.system_prompt_template = base_prompt_template.replace( '{functions}', tool_descriptions) if tool_descriptions else "No tools available."
            logger.debug(f"System prompt template set:\n{self.system_prompt_template[:300]}...")

        except httpx.RequestError as e:
            logger.error(f"HTTP error connecting to prompt server at {prompts_url}: {e}. Using default prompt.")
            # Format the default prompt if server fetch failed
            self.system_prompt_template = self.default_prompt.format(
                functions=tool_descriptions if tool_descriptions else "No tools available."
            )
        except httpx.HTTPStatusError as e:
             logger.error(f"HTTP status error {e.response.status_code} from prompt server {prompts_url}. Using default prompt.")
             self.system_prompt_template = self.default_prompt.format(
                functions=tool_descriptions if tool_descriptions else "No tools available."
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching prompt from server: {e}. Using default prompt.", exc_info=True)
            self.system_prompt_template = self.default_prompt.format(
                functions=tool_descriptions if tool_descriptions else "No tools available."
            )

    def _validate_schema(self, params_schema: Dict[str, Any]) -> bool:
        """Validate a tool schema (basic check)."""
        if not isinstance(params_schema, dict):
            logger.warning(f"Schema validation failed: not a dict ({type(params_schema)})")
            return False
        if params_schema.get('type') != 'object':
             logger.warning(f"Schema validation failed: 'type' is not 'object' ({params_schema.get('type')})")
             # Allow non-object schemas if properties is missing (e.g. tool takes no args)
             if 'properties' not in params_schema:
                 logger.debug("Schema allows no arguments (no 'properties').")
                 return True # Valid schema with no properties
             return False
        if 'properties' in params_schema and not isinstance(params_schema.get('properties'), dict):
            logger.warning(f"Schema validation failed: 'properties' is not a dict ({type(params_schema.get('properties'))})")
            return False
        # Check properties structure (optional, but good)
        for name, prop in params_schema.get('properties', {}).items():
            if not isinstance(prop, dict) or 'type' not in prop:
                logger.warning(f"Schema validation failed: Property '{name}' is invalid ({prop})")
                return False
        return True

    async def initialize_tools(self):
        """Fetch tool list via SSE and schemas, then initialize tools in the Models class."""
        tools_url = f"{self.config.server_base_url}/mcp/tools"
        logger.info(f"Fetching tool list from server via SSE: {tools_url}")
        server_tools_data = None

        try:
            async with aconnect_sse(self.http_client, "GET", tools_url) as event_source:
                async for sse in event_source.aiter_sse():
                    logger.debug(f"SSE Tools List Received: event='{sse.event}', data='{sse.data[:200]}...'")

                    if sse.event == "error":
                        error_message = "Unknown SSE error fetching tools"
                        if sse.data:
                            try:
                                error_data = json.loads(sse.data)
                                error_message = error_data.get("error", error_message)
                            except json.JSONDecodeError:
                                error_message = f"Non-JSON error data: {sse.data}"
                        logger.error(f"Server sent error event while fetching tools: {error_message}")
                        raise ConnectionError(f"Server error fetching tools list: {error_message}")

                    elif sse.event == "tools_list":
                        if not sse.data:
                            raise ConnectionError("Invalid empty data received for tools list.")
                        try:
                            data = json.loads(sse.data)
                            if "available_tools" not in data or not isinstance(data["available_tools"], list):
                                raise ConnectionError("Received invalid format for tools list from server via SSE.")

                            server_tools_data = data["available_tools"]
                            logger.info(f"Received {len(server_tools_data)} tool definitions from server via SSE.")
                            break

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON data for 'tools_list' event: {e}. Data: {sse.data}")
                            raise ConnectionError("Invalid JSON received in 'tools_list' event.") from e

            if server_tools_data is None:
                raise ConnectionError("Did not receive tool list from server via SSE.")

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError, httpx.HTTPStatusError, SSEError) as e:
            logger.error(f"Error connecting to or processing SSE stream for tools: {e}", exc_info=True)
            raise ConnectionError(f"Failed to fetch tools via SSE: {e}") from e
        except ConnectionError:
            raise  # Re-raise ConnectionErrors from within the try block
        except Exception as e:
            logger.error(f"Unexpected error fetching tools via SSE: {e}", exc_info=True)
            raise ConnectionError(f"Failed to fetch tools via SSE: {e}") from e


        self.mcp_tools_map = {}
        self.tools_schema_for_binding = []
        valid_tools_count = 0
        for tool in server_tools_data:
            tool_name = tool.get('name')
            if not tool_name:
                logger.warning(f"Skipping tool due to missing name: {tool}")
                continue
            if 'parameters' not in tool:
                logger.warning(f"Tool '{tool_name}' is missing 'parameters' field. Assuming no parameters.")
                tool['parameters'] = {'type': 'object', 'properties': {}}  # Provide default if missing

            if not self._validate_schema(tool['parameters']):
                logger.warning(f"Skipping tool '{tool_name}' due to invalid parameter schema: {tool.get('parameters')}")
                continue

            # Remove 'title' if present (FIX)
            if 'title' in tool:
                del tool['title']

            self.mcp_tools_map[tool_name] = tool
            self.tools_schema_for_binding.append({
                "name": tool_name,
                "description": tool.get('description', f"Tool named {tool_name}"),
                "parameters": tool['parameters']
            })
            valid_tools_count += 1

        logger.info(f"Validated {valid_tools_count} tools: {list(self.mcp_tools_map.keys())}")
        print(f"\nConnected to server. Validated Tools: {list(self.mcp_tools_map.keys())}")

        try:
            self.models.initializeTools(self.tools_schema_for_binding)
            logger.info(f"Successfully initialized {len(self.models.tool_schemas_for_binding)} tools in Models class.")
        except Exception as e:
                logger.error(f"Failed to initialize tools in Models class: {e}", exc_info=True)
                raise ConnectionError("Failed to process and bind tools to the LLM.") from e

        await self._prepare_system_prompt()


    # --- Caching Logic (Keep _should_bypass_cache and _generate_cache_key as they are) ---
    def _should_bypass_cache(self, query: str, tool_name: Optional[str] = None) -> bool:
        """Determine if caching should be bypassed for a query or tool."""
        if tool_name and tool_name in self.no_cache_tools:
            logger.debug(f"Bypassing cache for tool: {tool_name}")
            return True
        # Check query patterns only if tool_name doesn't force bypass
        if query and any(pattern.lower() in query.lower() for pattern in self.no_cache_query_patterns):
            logger.debug(f"Bypassing cache for query pattern match: {query[:100]}...")
            return True
        return False

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Ensure consistent ordering for kwargs
        key_str = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()


    # --- Tool Execution (_execute_mcp_tool_internal and _execute_mcp_tool use SSE correctly) ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _execute_mcp_tool_internal(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper for executing a tool via SSE with retries (no caching here)."""
        encoded_args = urllib.parse.quote(json.dumps(tool_args))
        tool_url = f"{self.config.server_base_url}/mcp/tools/{tool_name}?arguments={encoded_args}"
        logger.info(f"Executing tool '{tool_name}' via SSE: {tool_url}") # Log actual call

        final_result = None
        error_message = None
        status_messages = []

        try:
            # Correctly uses aconnect_sse for tool execution endpoint
            async with aconnect_sse(self.http_client, "GET", tool_url) as event_source:
                async for sse in event_source.aiter_sse():
                    logger.debug(f"SSE received: event='{sse.event}', data='{sse.data[:200]}...'")
                    data = None
                    if sse.data:
                        try:
                            data = json.loads(sse.data)
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON data for event '{sse.event}': {sse.data}")
                            data = {"raw_data": sse.data}
                    else:
                         data = {}


                    if sse.event == "error":
                        error_message = data.get("error", f"Unknown error from tool '{tool_name}'") if isinstance(data, dict) else f"Unknown error: {sse.data}"
                        logger.error(f"Tool '{tool_name}' reported error: {error_message}")
                        status_messages.append(f"ERROR_EVENT: {error_message}")
                        break # Stop processing on explicit error

                    elif sse.event == "status":
                         status_content = "Status update (no details)"
                         if isinstance(data, dict):
                             status_content = data.get("status", data.get("message", json.dumps(data)))
                         elif isinstance(data, str):
                              status_content = data
                         status_messages.append(status_content)
                         print(f"Status ({tool_name}): {status_content}") # Show status to user

                    elif sse.event == "final_result":
                         if isinstance(data, dict):
                             final_result = data
                             logger.info(f"Received final_result event for {tool_name}")
                         else:
                              logger.warning(f"Received non-dict data for final_result event: {data}. Treating as error.")
                              error_message = f"Invalid format for final_result: {sse.data}"
                         break # Stop processing on final result
                    else:
                         logger.debug(f"Received unexpected SSE event '{sse.event}': {sse.data}")

            # After loop finishes
            if error_message:
                 logger.error(f"Tool execution failed for '{tool_name}'. Error: {error_message}")
                 return {"error": error_message, "status_updates": status_messages}
            elif final_result is not None:
                 logger.info(f"Tool '{tool_name}' execution successful.")
                 final_result["status_updates_during_execution"] = status_messages
                 return final_result
            else:
                 logger.warning(f"Tool '{tool_name}' SSE stream ended without explicit 'final_result' or 'error' event.")
                 return {"status": "completed_without_explicit_result", "status_updates": status_messages}

        # --- Catch specific httpx errors ---
        except httpx.HTTPStatusError as e:
             logger.error(f"HTTP status error {e.response.status_code} during SSE connection for tool '{tool_name}': {e.response.text[:500]}", exc_info=True)
             raise ToolExecutionError(f"Server error {e.response.status_code} executing tool '{tool_name}'.") from e
        except httpx.ReadTimeout:
             logger.error(f"Read timeout waiting for SSE events from tool '{tool_name}'", exc_info=True)
             raise ToolExecutionError(f"Timeout waiting for response from tool '{tool_name}'")
        except httpx.RequestError as e:
             logger.error(f"Network error during SSE connection for tool '{tool_name}': {e}", exc_info=True)
             raise ToolExecutionError(f"Network error connecting to tool '{tool_name}': {str(e)}") from e
        # --- Catch SSE processing errors ---
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON from SSE stream for tool '{tool_name}': {e}", exc_info=True)
             raise ToolExecutionError(f"Invalid JSON received from tool '{tool_name}'.") from e
        except SSEError as e: # Catch errors specific to httpx_sse processing
             logger.error(f"SSE processing error for tool '{tool_name}': {e}", exc_info=True)
             raise ToolExecutionError(f"Error processing SSE stream for tool '{tool_name}': {str(e)}") from e
        except Exception as e:
             logger.error(f"Unexpected error processing SSE stream for tool '{tool_name}': {e}", exc_info=True)
             raise ToolExecutionError(f"Error processing tool '{tool_name}' response: {str(e)}") from e


    async def _execute_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a tool via the server's SSE endpoint with caching and retries."""
        if tool_name not in self.mcp_tools_map:
            logger.error(f"Attempted to call unknown or invalid tool: {tool_name}")
            return {"error": f"Tool '{tool_name}' is not available or has an invalid schema."}

        # --- Caching Logic ---
        should_bypass = self._should_bypass_cache("", tool_name) # Base bypass on tool name
        cache_key = self._generate_cache_key(tool_name, tool_args)

        if not should_bypass:
             cached_result = self.tool_cache.get(cache_key)
             if cached_result is not None:
                 logger.info(f"Tool cache hit for {tool_name} with args {tool_args}")
                 # Return a copy to prevent modification of cached object
                 return cached_result.copy()
             else:
                  logger.info(f"Tool cache miss for {tool_name} with args {tool_args}")
        else:
             logger.info(f"Tool cache bypassed for {tool_name}")

        # --- Execute (Cache Miss or Bypass) ---
        try:
             result = await self._execute_mcp_tool_internal(tool_name, tool_args)

             # Cache the result ONLY if it's not an error and caching wasn't bypassed
             if not should_bypass and isinstance(result, dict) and 'error' not in result:
                  self.tool_cache[cache_key] = result.copy() # Cache a copy
                  logger.debug(f"Stored result for {tool_name} in cache.")

             return result

        except ToolExecutionError as e:
             logger.error(f"Tool execution failed for '{tool_name}' after retries: {e}")
             # Return the error structure
             return {"error": str(e)}
        except Exception as e:
             # Catch unexpected errors from the retry mechanism itself or other issues
             logger.error(f"Unexpected error during tool execution wrapper for '{tool_name}': {e}", exc_info=True)
             return {"error": f"Unexpected error executing tool '{tool_name}': {str(e)}"}


    # --- LLM Invocation (Needs update for Google API) ---
    async def _invoke_llm(self, messages: List[Any], request_structured_output: bool = False, output_schema=None) -> Any:
        """
        Invoke the Google LLM with appropriate formatting and optional structured output.
        Returns the raw Google response object or the parsed Pydantic object.
        """
        latest_query_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        system_prompt_content = next((m.content for m in messages if isinstance(m, SystemMessage)), "") # Assuming one system prompt

        # --- Caching Logic (Applied before calling the API) ---
        cached_response = None # Initialize here
        should_bypass = self._should_bypass_cache(latest_query_content) # Check query patterns
        cache_key = None
        if not should_bypass and not request_structured_output: # Only cache simple invocations
             # Generate a more robust cache key including message history relevant parts
             history_key_part = json.dumps([{'role': type(m).__name__, 'content': str(m.content)[:100]} for m in messages[-5:]], sort_keys=True) # Key based on last few messages
             cache_key = self._generate_cache_key(system_prompt_content, history_key_part)
             cached_response = self.llm_cache.get(cache_key)
             if cached_response is not None:
                 logger.info("LLM cache hit.")
                 # Ensure cached response is the expected type (Google Response)
                 if isinstance(cached_response, google_types.GenerateContentResponse):
                     return cached_response
                 else:
                      logger.warning("LLM cache contained unexpected type. Ignoring cache.")
                      cached_response = None # Reset to trigger actual call

        logger.info(f"Invoking LLM ({'structured' if request_structured_output else 'general'}). Cache: {'miss' if not cached_response else 'hit (ignored)'} / {'bypassed' if should_bypass else 'enabled'}.")

        try:
             if request_structured_output:
                 if not output_schema:
                     raise ValueError("output_schema must be provided for structured output request.")
                 # Call the structured invocation method in Models
                 response = await self.models.invoke_structured_plan(messages) # Assumes invoke_structured_plan uses the right messages
                 # Caching for structured output is complex, skipping for now.
                 return response # Returns the Pydantic Plan object
             else:
                 # Call the general invocation method in Models
                 response = ic(await self.models.invoke_prompt(messages))
                 # Cache the raw Google response if caching enabled and successful
                 if cache_key and isinstance(response, VertexChatMessage):
                      # Check for blocked content before caching
                      if not response.text or (hasattr(response, 'finish_reason') and response.finish_reason != 'STOP'):
                           logger.warning(f"LLM response may be blocked or incomplete (reason: {response.finish_reason if hasattr(response, 'finish_reason') else 'N/A'}). Not caching.")
                      else:
                           self.llm_cache[cache_key] = response
                           logger.debug("Stored LLM response in cache.")
                 return response # Returns the Google response object

        except (RuntimeError, ValueError, ConnectionError) as e:
             # Catch errors from Models methods or API calls
             logger.error(f"LLM invocation failed: {e}", exc_info=True)
             # Re-raise to be handled by the calling function (process_query or _execute_plan)
             raise
        except Exception as e:
            # Catch unexpected errors
            logger.error(f"Unexpected error during LLM invocation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected LLM error: {e}") from e


    def _format_tool_result(self, tool_result: Any, tool_name: str) -> str:
        """Format tool result for readable output and history."""
        if isinstance(tool_result, dict) and 'error' in tool_result:
            # Include status updates if available in the error dict
            statuses = tool_result.get('status_updates', [])
            status_str = f"\nStatus Updates:\n" + "\n".join(statuses) if statuses else ""
            return f"Error from {tool_name}: {tool_result['error']}{status_str}"
        elif isinstance(tool_result, dict):
            # Try pretty-printing JSON, fallback to string
            try:
                # Exclude status updates from the main JSON output for clarity
                result_data = {k: v for k, v in tool_result.items() if k != "status_updates_during_execution"}
                if not result_data: # If only statuses were returned
                     statuses = tool_result.get("status_updates_during_execution", [])
                     return f"Tool {tool_name} completed.\nStatus Updates:\n" + "\n".join(statuses) if statuses else f"Tool {tool_name} completed with no specific output."

                readable_response = json.dumps(result_data, indent=2)
                # Append statuses separately if they exist
                statuses = tool_result.get("status_updates_during_execution", [])
                if statuses:
                     readable_response += "\n\nStatus Updates Received:\n" + "\n".join(statuses)

            except Exception as e:
                logger.warning(f"Could not format dict result for {tool_name} as JSON: {e}")
                readable_response = str(tool_result) # Fallback to plain string
        elif isinstance(tool_result, list):
            # Format lists nicely, limiting length
            limit = 50
            item_strs = [str(item) for item in tool_result[:limit]]
            readable_response = "\n".join(item_strs)
            if len(tool_result) > limit:
                readable_response += f"\n\n... (truncated, {len(tool_result) - limit} more items)"
        else:
            # Handle strings, numbers, etc.
            readable_response = str(tool_result)
            if not readable_response:
                 readable_response = f"Tool {tool_name} executed successfully, no output provided."

        # Limit overall length for history/display
        max_len = 5000 # Adjust as needed
        if len(readable_response) > max_len:
             readable_response = readable_response[:max_len] + f"... (tool result truncated at {max_len} chars)"

        return readable_response

    # --- Plan Execution Logic (Looks okay, handles Google Response structure) ---
    async def _execute_plan(self, ai_plan: Plan, executed_tool_calls: set) -> str:
        """Executes steps defined in a Plan object."""
        if not isinstance(ai_plan, Plan) or not hasattr(ai_plan, 'plan') or not isinstance(ai_plan.plan, list):
             logger.error(f"Invalid plan structure received: {ai_plan}")
             # Append an error message to history?
             self.conversation_history.append(AIMessage(content="(Error: Invalid plan structure received from LLM. Cannot execute.)"))
             return "(Error: Invalid plan structure received from LLM)"

        final_plan_outcome = "" # Store the final result/summary of the plan execution
        plan_failed = False # Flag to indicate if plan execution should stop

        try:
            for i, step in enumerate(ai_plan.plan):
                if plan_failed: # Stop processing steps if a failure occurred and we decided to halt
                    logger.warning(f"Skipping remaining plan steps due to previous failure.")
                    break

                logger.info(f"--- Executing Plan Step {i+1}/{len(ai_plan.plan)} ---")
                if not isinstance(step, ToolStep):
                     logger.error(f"Invalid step type in plan: {type(step)}. Step: {step}")
                     final_plan_outcome += f"\n(Error: Skipping invalid step {i+1})"
                     continue

                # --- Nested Plan ---
                if step.isPlan:
                    logger.info(f"Step {i+1}: Executing nested plan with prompt: '{step.prompt[:100]}...'")
                    print(f"\nAssistant (Nested Plan): {step.prompt}") # User feedback

                    # Add prompt to history, invoke structured plan again
                    self.conversation_history.append(HumanMessage(content=step.prompt))
                    await self._summarize_history()
                    self._truncate_history()

                    try:
                        # Use the structured LLM call directly
                        nested_plan_obj: Plan = await self._invoke_llm(
                            self.conversation_history,
                            request_structured_output=True,
                            output_schema=Plan
                        )
                        # Recursive call with the new Plan object
                        nested_results = await self._execute_plan(nested_plan_obj, executed_tool_calls) # Pass the same set
                        final_plan_outcome += f"\nNested Plan ({step.prompt[:30]}...) Result:\n{nested_results}"
                        # Check if nested execution returned an error message indicating failure
                        if nested_results.startswith("(Error:") or nested_results.startswith("Plan Execution Summary Failed"):
                            logger.warning(f"Nested plan for step {i+1} failed.")
                            # Decide whether to stop outer plan
                            # plan_failed = True
                        # History (like AIMessage for the plan, tool calls/results) is managed within the recursive call
                    except (RuntimeError, ValueError, ConnectionError) as e:
                         logger.error(f"Failed to generate or execute nested plan for step {i+1}: {e}", exc_info=True)
                         error_msg = f"(Error generating/executing nested plan for step '{step.prompt[:50]}...': {e})"
                         self.conversation_history.append(AIMessage(content=error_msg))
                         final_plan_outcome += f"\n{error_msg}"
                         plan_failed = True # Stop outer plan execution on nested failure

                    continue # Move to the next step in the current level

                # --- Prompt Step (leading to potential tool call or direct answer) ---
                elif step.prompt:
                    current_prompt = step.prompt
                    print(f"\nAssistant (Executing Step): {current_prompt}") # User feedback
                    logger.info(f"Step {i+1}: Executing prompt: '{current_prompt[:100]}...'")

                    # Add the step's prompt to history
                    self.conversation_history.append(HumanMessage(content=current_prompt))
                    await self._summarize_history()
                    self._truncate_history()

                    # Invoke LLM for this specific prompt step (general invocation, might return tool calls)
                    try:
                        # Use the general LLM call
                        google_response: google_types.GenerateContentResponse = await self._invoke_llm(
                             self.conversation_history
                        )

                        # --- Process Google Response ---
                        if not google_response.candidates:
                             logger.error(f"LLM response has no candidates for step {i+1}.")
                             step_result_msg = "(Error: LLM response was empty or invalid)"
                             self.conversation_history.append(AIMessage(content=step_result_msg))
                             final_plan_outcome += f"\nStep {i+1} Failed: {step_result_msg}"
                             plan_failed = True # Consider this a failure
                             continue # Try next step (or stop if plan_failed)

                        # Check for safety blocking first
                        candidate = google_response.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN') # Safely get finish_reason
                        if finish_reason != 'STOP': # Check for non-successful termination
                             logger.warning(f"LLM response for step {i+1} finished with reason: {finish_reason}.")
                             # Extract safety ratings if available
                             safety_message = ""
                             if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                 safety_message = f" (Safety: {candidate.safety_ratings})"
                             step_result_msg = f"(Assistant response may be incomplete or blocked due to reason: {finish_reason}{safety_message})"
                             self.conversation_history.append(AIMessage(content=step_result_msg))
                             final_plan_outcome += f"\nStep {i+1} Issue: {step_result_msg}"
                             # Decide if this is fatal (e.g., SAFETY often is)
                             if finish_reason == 'SAFETY':
                                 plan_failed = True
                             # Continue processing content if any, but flag potential issue

                        # Check for tool calls (FunctionCall)
                        tool_calls_in_response = []
                        ai_message_content = ""
                        if candidate.content and candidate.content.parts:
                             for part in candidate.content.parts:
                                 if hasattr(part, 'function_call') and part.function_call: # Check attribute exists
                                     tool_calls_in_response.append(part.function_call)
                                     logger.info(f"Step {i+1}: LLM requested tool call: {part.function_call.name}")
                                 elif hasattr(part, 'text'): # Check attribute exists
                                     ai_message_content += part.text # Aggregate text parts

                        # Construct LangChain AIMessage for history
                        lc_tool_calls = []
                        if tool_calls_in_response:
                           for idx, func_call in enumerate(tool_calls_in_response):
                               # Generate a temporary ID (e.g., based on name and index + hash)
                               temp_tool_call_id = f"{func_call.name}_{idx}_{hashlib.md5(str(func_call.args).encode()).hexdigest()[:8]}"
                               lc_tool_calls.append({
                                   "id": temp_tool_call_id,
                                   "name": func_call.name,
                                   "args": dict(func_call.args) # Convert Struct/dict-like to dict
                               })

                        # Ensure content is a string, handle potential None or other types gracefully
                        ai_msg_for_history = AIMessage(
                            content=str(ai_message_content) if ai_message_content is not None else "",
                            tool_calls=lc_tool_calls if lc_tool_calls else None # Use None if empty list
                        )
                        self.conversation_history.append(ai_msg_for_history)
                        self._truncate_history() # Truncate after adding AI response

                        # --- Tool Call Handling ---
                        if tool_calls_in_response:
                            logger.info(f"Step {i+1}: Processing {len(tool_calls_in_response)} tool call(s): {[tc.name for tc in tool_calls_in_response]}")
                            tool_results_for_step = [] # Collect results for this step
                            step_had_tool_error = False # Track errors within this step

                            for lc_tool_call in lc_tool_calls: # Iterate through our generated LC format calls
                                tool_name = lc_tool_call["name"]
                                tool_args = lc_tool_call["args"]
                                tool_call_id = lc_tool_call["id"] # Use the temp ID we generated

                                if not isinstance(tool_args, dict):
                                    logger.error(f"Malformed tool args for {tool_name}: {tool_args}")
                                    tool_result_content = json.dumps({"error": f"Internal error: Invalid arguments format for {tool_name}."})
                                    tool_result = {"error": f"Internal error: Invalid arguments format for {tool_name}."}
                                    # Don't add to executed_tool_calls if args are bad
                                else:
                                    # Check for redundant calls
                                    tool_call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
                                    if tool_call_key in executed_tool_calls:
                                        logger.warning(f"Step {i+1}: Skipping redundant tool call: {tool_name} with args {tool_args}")
                                        tool_result_content = json.dumps({"status": "skipped_redundant", "message": f"Tool call {tool_name} with these arguments was already executed."})
                                        tool_result = {"status": "skipped_redundant", "message": f"Tool call {tool_name} with these arguments was already executed."}
                                    else:
                                        # Execute the tool
                                        tool_result = await self._execute_mcp_tool(tool_name, tool_args)
                                        self.last_tool_result = tool_result # Store raw result
                                        tool_result_content = json.dumps(tool_result) # Serialize for history
                                        executed_tool_calls.add(tool_call_key) # Add executed call key

                                # Append ToolMessage to history
                                tool_message = ToolMessage(
                                    content=tool_result_content,
                                    tool_call_id=tool_call_id # Match the ID from AIMessage
                                )
                                self.conversation_history.append(tool_message)
                                self._truncate_history() # Truncate after adding tool result
                                formatted_result = self._format_tool_result(tool_result, tool_name)
                                tool_results_for_step.append(formatted_result) # Add formatted result

                                # Handle tool error propagation
                                if isinstance(tool_result, dict) and 'error' in tool_result:
                                    error_msg = f"Tool '{tool_name}' failed: {tool_result['error']}"
                                    logger.error(f"Step {i+1}: {error_msg}")
                                    step_had_tool_error = True
                                    # Decide if you want to stop the whole plan on tool error
                                    # plan_failed = True # Uncomment to stop entire plan on any tool error

                            # Combine results for the step outcome
                            step_outcome = "\n".join(tool_results_for_step)
                            if ai_message_content: # Include AI text if present with tool calls
                                step_outcome = f"{ai_message_content}\n{step_outcome}"
                            final_plan_outcome += f"\nStep {i+1} Result:\n{step_outcome}"
                            print(f"Assistant (Tool Results):\n{step_outcome}") # Show tool results

                            if step_had_tool_error and plan_failed: # If we decided to stop plan on tool error
                                logger.error(f"Plan execution stopped due to tool error in step {i+1}.")
                                # Optionally return early or just let the loop break next iteration
                                # return f"(Plan execution stopped due to error in step {i+1}: {current_prompt})"


                        # --- No Tool Call - Direct LLM Response ---
                        elif ai_message_content or finish_reason == 'STOP': # If there's content or it finished normally without tools
                            logger.info(f"Step {i+1}: No tool calls requested. Using LLM content.")
                            step_outcome = ai_message_content or f"(Step {i+1} completed with no text output)"
                            print(f"Assistant: {step_outcome}") # Show intermediate result
                            final_plan_outcome += f"\nStep {i+1} Result:\n{step_outcome}"
                        # Else: If finish_reason wasn't STOP and no content/tools, the warning above handles it.


                    except (RuntimeError, ValueError, ConnectionError) as e:
                        # Catch LLM invocation errors for this step
                        logger.error(f"LLM invocation failed for step {i+1} ('{current_prompt[:50]}...'): {e}", exc_info=True)
                        error_msg = f"(Error invoking LLM for step '{current_prompt[:50]}...': {e})"
                        self.conversation_history.append(AIMessage(content=error_msg)) # Add error to history
                        final_plan_outcome += f"\nStep {i+1} Failed: {error_msg}"
                        plan_failed = True # Stop plan execution on LLM error within a step

                # --- Invalid Step Structure ---
                else:
                    logger.error(f"Step {i+1}: Invalid step structure (missing prompt and not a plan): {step}")
                    final_plan_outcome += f"\n(Error: Skipping invalid step {i+1} - missing prompt)"

            # --- Plan Execution Complete ---
            logger.info("--- Plan Execution Finished ---")

            # Check if the plan failed at any point
            if plan_failed:
                 logger.error("Plan execution failed at some step.")
                 # Return the accumulated outcome, which should contain error messages
                 return f"(Plan execution failed. See logs and previous messages for details)\nAccumulated Outcome:\n{final_plan_outcome}"

            # If successful, make a final LLM call to summarize the execution
            final_summary_prompt = f"The following plan steps were executed successfully:\n{final_plan_outcome}\n\nProvide a final, concise summary response to the original user query based *only* on these results. Do not mention the plan steps themselves unless the result was an error."
            self.conversation_history.append(HumanMessage(content="[System Note] Summarize the plan execution results for the user.")) # Internal marker
            self.conversation_history.append(HumanMessage(content=final_summary_prompt))
            self._truncate_history()
            try:
                 final_response_obj = ic(await self._invoke_llm(self.conversation_history))
                 final_summary = final_response_obj.text or "(Plan executed, but failed to get final summary.)"
                 # Check summary for safety issues
                 if final_response_obj.candidates and final_response_obj.candidates[0].finish_reason != 'STOP':
                     logger.warning(f"Final summary generation finished with reason: {final_response_obj.candidates[0].finish_reason}")
                     final_summary = f"(Summary generation issue: {final_response_obj.candidates[0].finish_reason}). Raw Outcome:\n{final_plan_outcome}"

                 self.conversation_history.append(AIMessage(content=final_summary)) # Add summary to history
                 self._truncate_history()
                 return final_summary
            except Exception as e:
                 logger.error(f"Failed to get final summary after plan execution: {e}", exc_info=True)
                 # Return the raw accumulated outcome if summary fails
                 return f"Plan Execution Summary Failed. Raw Outcome:\n{final_plan_outcome}"


        except Exception as e:
            logger.error(f"Unexpected error during plan execution loop: {e}", exc_info=True)
            if "rate limit" in str(e).lower():
                return "(Error: API Rate Limit Exceeded during plan execution.)"
            # Add error message to history?
            error_msg = f"(Critical Error during plan execution: {e})"
            self.conversation_history.append(AIMessage(content=error_msg))
            return f"Error executing plan: {str(e)}\nAccumulated Outcome:\n{final_plan_outcome}"


    async def process_query(self, query: str) -> str:
        """Process a user query: generate plan, execute plan."""
        if not query.strip():
            return "Query cannot be empty"

        logger.info(f"Processing query: {query[:100]}...")

        # 1. Initialize history with System Prompt if needed
        if not self.conversation_history or not isinstance(self.conversation_history[0], SystemMessage):
            if not self.system_prompt_template:
                await self._prepare_system_prompt()
                if not self.system_prompt_template:
                    logger.error("System prompt template could not be loaded.")
                    return "Error: Failed to load system prompt."
            # Prepend system message if missing
            self.conversation_history.insert(0, SystemMessage(content=self.system_prompt_template))


        # 2. Add User Query
        self.conversation_history.append(HumanMessage(content=query))
        # Summarize/Truncate *before* planning call if history is already long
        await self._summarize_history()
        self._truncate_history()

        # 3. Generate Plan
        logger.info("Generating execution plan...")
        try:
            ai_plan_obj: Plan = await self._invoke_llm(
                 self.conversation_history, # Send current history for planning context
                 request_structured_output=True,
                 output_schema=Plan
            )
            # Log the plan structure using Pydantic's dump for readability
            logger.info(f"Generated plan with {len(ai_plan_obj.plan)} steps.")
            ic(ai_plan_obj.model_dump()) # ic uses model_dump() by default on Pydantic models

        except (RuntimeError, ValueError, ConnectionError) as e:
             logger.error(f"Error generating execution plan: {e}", exc_info=True)
             # Check for specific errors like API key issues
             if "API key not valid" in str(e):
                 return "(Error: Invalid Google API Key. Please check configuration.)"
             return f"Error creating execution plan: {e}" # Return the specific error message
        except Exception as e:
             logger.error(f"Unexpected error during plan generation: {e}", exc_info=True)
             return f"Unexpected error generating plan: {str(e)}"


        # 4. Execute Plan
        logger.info("Executing generated plan...")
        try:
            executed_tool_calls = set() # Track executed calls across the plan
            final_result = await self._execute_plan(ai_plan_obj, executed_tool_calls)
            # The final result from _execute_plan should be the user-facing response
            return final_result
        # except MaxToolIterationsError as e: # Keep if _execute_plan doesn't handle it internally
        #      logger.error(f"Max tool iterations reached: {e}")
        #      return f"\nError: {e}"
        except Exception as e:
            logger.error(f"Error during plan execution phase: {e}", exc_info=True)
            if "rate limit" in str(e).lower():
                return "(Error: API Rate Limit Exceeded during plan execution.)"
            return f"Error executing plan: {str(e)}"


    async def chat_loop(self):
        """Run an interactive chat loop with conversation context."""
        # Check if Models object exists, not specific LLM instance
        if not self.models:
            logger.error("Cannot start chat: Models object not initialized.")
            print("\nError: Could not initialize models. Exiting.")
            return
        # Check if tools were loaded (mcp_tools_map)
        if not self.mcp_tools_map:
             logger.warning("Starting chat loop, but no tools were loaded from the server.")
             print("\nWarning: No tools loaded from the server. Functionality may be limited.")
             # Continue anyway, LLM might work without tools

        logger.info("Starting interactive chat loop (SSE Client - Google Gemini)")
        print("\nMCP SSE Client (Google Gemini) Started!")
        print(f"Connected to: {self.config.server_base_url}")
        print(f"Using Tools: {list(self.mcp_tools_map.keys()) if self.mcp_tools_map else 'None'}") # Handle empty map
        print("Conversation context is maintained.")
        print("Type your queries, 'quit' to exit, or 'clear' to reset history.")

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
                print("\nAssistant (thinking... generating plan... executing...)") # Thinking indicator
                response = await self.process_query(query)

                # Print final response (should be string from process_query)
                print("\nAssistant:")
                if isinstance(response, str):
                    print(response)
                else:
                    # Should not happen if process_query returns string, but fallback
                    logger.error(f"process_query returned unexpected type: {type(response)}")
                    print(str(response))

            except KeyboardInterrupt:
                logger.warning("Chat loop interrupted by user (Ctrl+C)")
                print("\nExiting chat loop...")
                break
            # Keep MaxToolIterationsError if relevant
            # except MaxToolIterationsError as e:
            #     logger.error(f"Max tool iterations reached: {e}")
            #     print(f"\nError: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in chat loop: {e}", exc_info=True)
                print(f"\nAn unexpected error occurred: {str(e)}")
                print("Type 'quit' to exit, 'clear' to reset history, or try another query.")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up MCP SSE client resources")
        try:
            if self.http_client and not self.http_client.is_closed:
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
        # Initialization order: Config -> Client -> Initialize Tools (which inits Models tools) -> Prepare Prompt
        client = MCPClient()
        # **** Initialize tools MUST be awaited ****
        await client.initialize_tools() # This now fetches tools via SSE and initializes them in Models
        # System prompt prepared at the end of initialize_tools
        await client.chat_loop()
    except ConfigError as e:
        logger.critical(f"Configuration error: {e}")
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file (MCP_SERVER_BASE_URL, GOOGLE_API_KEY, DEFAULT_PROMPT_TEMPLATE).")
    except ConnectionError as e:
        # Catch errors from initialize_tools (SSE/connection/parsing)
        logger.critical(f"Connection or Initialization error: {e}")
        print(f"\nConnection/Initialization Error: {e}")
        server_url = 'N/A'
        if client and hasattr(client, 'config') and client.config:
             server_url = client.config.server_base_url
        print(f"Could not connect to the MCP server at {server_url}, retrieve/initialize tools via SSE, or communicate with Google API.")
        print("Ensure the MCP FastAPI server is running and accessible. Check server logs and network.")
        print("Also check your Google API key validity.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if client:
            await client.cleanup()
        logger.info("--- MCP SSE Client Application Shutdown Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    # Add PYTHONASYNCIODEBUG=1 for more detailed async debugging if needed
    # os.environ['PYTHONASYNCIODEBUG'] = '1'
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        logger.warning("Application terminated by user (KeyboardInterrupt).")
    except Exception as e:
        # Catch errors that might occur during asyncio.run() itself or final cleanup
        print(f"\nFatal error during application execution or shutdown: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)

# --- END OF FILE main.py ---
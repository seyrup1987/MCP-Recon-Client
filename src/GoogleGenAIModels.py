import os
import json
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Tool,
    ToolConfig
)
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
from icecream import ic

# Import LangChain message types for conversion
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Get logger for this module
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables early for API key access

def convert_json_a_to_b(json_a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts JSON object A to the structure of JSON object B.
    
    Args:
        json_a (dict): Input JSON object A
        
    Returns:
        dict: Converted JSON object in the structure of B
    """
    json_b = {
        "name": json_a.get("name", ""),
        "description": json_a.get("description", "").strip(),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    if "parameters" in json_a and "properties" in json_a["parameters"]:
        properties_a = json_a["parameters"]["properties"]
        json_b["parameters"]["properties"] = {
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"List derived from library: {properties_a.get('library_name', {}).get('title', '')}"
            },
            "date": {
                "type": "string",
                "description": f"Date associated with source: {properties_a.get('source_url', {}).get('title', '')}"
            },
            "time": {
                "type": "string",
                "description": f"Time related to language: {properties_a.get('language', {}).get('title', '')}"
            },
            "topic": {
                "type": "string",
                "description": f"Topic from documentation: {properties_a.get('documentation_text', {}).get('title', '')}"
            }
        }
        json_b["parameters"]["required"] = ["attendees", "date", "time", "topic"]
    
    return json_b

class ToolStep(BaseModel):
    prompt: str = Field(..., description="Prompt or instruction for the step")
    isPlan: bool = Field(..., description="Whether this step represents a nested plan")

class Plan(BaseModel):
    """Defines the structure for the generated plan, which is a list of ToolSteps."""
    plan: List[ToolStep] = Field(..., description="An array of ToolSteps to execute in sequence")

class Models:
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
        LOCATION = os.getenv("GOOGLE_LOCATION")

        if not google_api_key:
            error_msg = "GOOGLE_API_KEY must be set in environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Configure genai with the API key
            self.model_name = "gemini-1.5-flash"
            self.llm = genai.Client(
                api_key = google_api_key
            )
            logger.info(f"Initialized Google Generative AI Model: {self.model_name}")

            # Response schema for structured output
            self.response_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "isPlan": {"type": "boolean"}
                    },
                    "required": ["prompt", "isPlan"]
                }
            }
            self.tool_schemas_for_binding: List[Dict[str, Any]] = []
            self.tools: Optional[List[Tool]] = None

        except Exception as e:
            logger.exception(f"Failed to initialize Google Generative AI: {e}")
            raise

    def _format_messages_for_genai(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        Converts LangChain messages to google-generativeai's expected format.
        Ensures parts are valid and prevents misinterpretation as file-based inputs.
        """
        genai_messages = []
        for msg in messages:
            try:
                if isinstance(msg, SystemMessage):
                    genai_messages.append({
                        "role": "user",
                        "parts": [{"text": f"[SYSTEM] {msg.content}"}] if isinstance(msg.content, str) else []
                    })
                elif isinstance(msg, HumanMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    genai_messages.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
                elif isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if not isinstance(tool_call, dict) or 'name' not in tool_call or 'args' not in tool_call:
                                logger.warning(f"Skipping invalid tool call in AIMessage: {tool_call}")
                                continue
                            genai_messages.append({
                                "role": "model",
                                "parts": [{
                                    "function_call": {
                                        "name": tool_call['name'],
                                        "args": tool_call['args']
                                    }
                                }]
                            })
                    else:
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        genai_messages.append({
                            "role": "model",
                            "parts": [{"text": content}]
                        })
                elif isinstance(msg, ToolMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    genai_messages.append({
                        "role": "function",
                        "parts": [{
                            "function_response": {
                                "name": msg.tool_call_id,
                                "response": {"result": content}
                            }
                        }]
                    })
                else:
                    logger.warning(f"Skipping unsupported message type: {type(msg)}")
            except Exception as e:
                logger.error(f"Error formatting message {msg}: {e}")
                continue
        return genai_messages

    def initializeTools(self, toolsList: List[Dict[str, Any]]):
        """Stores tool schemas and creates the google-generativeai Tool objects."""
        if not toolsList:
            logger.warning("Received empty toolsList in initializeTools.")
            self.tools = None
            self.tool_schemas_for_binding = []
            return

        try:
            function_declarations = []
            self.tool_schemas_for_binding = toolsList

            for tool_schema in toolsList:
                converted_schema = convert_json_a_to_b(tool_schema)
                name = converted_schema.get("name")
                description = converted_schema.get("description")
                parameters = converted_schema.get("parameters")

                if not name or not description or not parameters:
                    logger.warning(f"Skipping tool due to missing name, description, or parameters: {tool_schema}")
                    continue

                function_declarations.append(
                    FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=parameters
                    )
                )

            self.tools = [Tool(function_declarations=function_declarations)]
            logger.info(f"Created Google Generative AI Tool object with {len(function_declarations)} function declarations.")

        except Exception as e:
            logger.exception(f"Failed to create Tool object from schemas: {e}")
            self.tools = None

    async def invoke_structured_plan(self, messages: List[Any]):
        """Invokes the LLM requesting a structured output conforming to the Plan schema."""
        last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_human_message:
            raise ValueError("Cannot generate plan without a user query (HumanMessage).")

        genai_messages = self._format_messages_for_genai(messages)
        if not genai_messages:
            logger.error("No valid messages after formatting for Google Generative AI.")
            raise ValueError("No valid messages to send to the LLM.")

        try:
            logger.debug(f"Invoking Google Generative AI for structured plan. Messages: {genai_messages}")

            config = GenerateContentConfig(
                tools=self.tools,
                temperature=0,
                responseMimeType="application/json",
                responseSchema=self.response_schema
            )

            config.tool_config = ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY
                )
            )

            response = ic(self.llm.models.generate_content(
                model=self.model_name,
                contents=genai_messages,  # Use formatted messages
                config=config
            ))

            logger.info("Structured LLM invoked successfully for plan generation.")
            logger.debug(f"Raw Response: {response.text}")

            if not response.text:
                logger.error("No text content received from Google Generative AI.")
                raise RuntimeError("LLM response invalid: No text content.")

            try:
                plan_data = json.loads(response.text)
                validated_plan = Plan(plan=[ToolStep(**step) for step in plan_data])
                logger.debug(f"Received and validated Plan structure: {validated_plan.model_dump()}")
                return validated_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}. Raw text: {response.text}")
                raise
            except ValidationError as e:
                logger.error(f"Failed to validate Plan structure: {e}. Raw JSON: {response.text}")
                raise

        except ValueError as e:
            logger.error(f"Invalid input to Google Generative AI: {e}")
            raise
        except Exception as e:
            logger.error(f"Error invoking structured LLM or processing response: {e}", exc_info=True)
            raise

    async def invoke_prompt(self, messages: List[Any]) -> genai.types.GenerateContentResponse:
        genai_messages = self._format_messages_for_genai(messages)
        if not genai_messages:
            logger.warning("invoke_prompt called with no valid messages after formatting.")
            raise ValueError("No valid messages to send to the LLM.")

        try:
            logger.debug(f"Invoking Google Generative AI for general prompt. Messages: {genai_messages}")

            config = GenerateContentConfig(
                tools=self.tools,
                temperature=0
            )

            config.tool_config = ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.AUTO
                )
            )

            response = ic(self.llm.models.generate_content(
                model=self.model_name,
                contents=genai_messages,  # Use formatted messages
                config=config
            ))

            logger.debug(f"Raw Response: {response}")
            return response

        except Exception as e:
            logger.error(f"Error occurred when processing prompt: {e}", exc_info=True)
            raise
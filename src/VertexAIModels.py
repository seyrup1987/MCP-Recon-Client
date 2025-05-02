import os
import json
# Use Pydantic v2 imports
from pydantic import BaseModel, Field, ValidationError
from vertexai.preview.language_models import ChatModel, CodeChatModel, InputOutputTextPair, ChatMessage, FunctionCall
from google.cloud import aiplatform
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
from icecream import ic

# Import LangChain message types for conversion
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Get logger for this module
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables early for API key access

def convert_json_a_to_b(json_a):
    """
    Converts JSON object A to the structure of JSON object B.
    
    Args:
        json_a (dict): Input JSON object A
        
    Returns:
        dict: Converted JSON object in the structure of B
    """
    # Initialize the output JSON B structure
    json_b = {
        "name": json_a.get("name", ""),
        "description": json_a.get("description", "").strip(),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    # Map parameters from A to B's structure
    if "parameters" in json_a and "properties" in json_a["parameters"]:
        properties_a = json_a["parameters"]["properties"]
        
        # Convert parameters to B's format
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
        
        # Set required fields (same as B's structure)
        json_b["parameters"]["required"] = ["attendees", "date", "time", "topic"]
    
    return json_b

class ToolStep(BaseModel):
    prompt: str = Field(...)
    isPlan: bool = Field(...)

class Plan(BaseModel):
    """Defines the overall structure for the generated plan, which is a list of ToolSteps."""
    plan: List[ToolStep] = Field(..., description="An array of ToolSteps to execute in sequence to fulfill the user's request.")


class Models:
    def __init__(self, model_name="chat-bison"):  # Use a valid Vertex AI model name
        google_api_key = os.getenv("GOOGLE_API_KEY")
        project_id = os.getenv("PROJECT_ID")  # Get project ID from environment
        location = os.getenv("LOCATION", "us-central1")  # Get location, default to us-central1

        if not google_api_key or not project_id:
            error_msg = "GOOGLE_API_KEY and PROJECT_ID must be set in environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.responseSchema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": ["STRING", "None"]},
                    "isPlan": {"type": ["BOOL"]}
                },
                "required": ["prompt", "isPlan"] # Explicitly stating required fields for the LLM
            }
        }

        try:
            # Initialize Vertex AI client
            aiplatform.init(project=project_id, location=location)
            self.model_name = model_name
            self.chat_model = ChatModel.from_pretrained(self.model_name)
            logger.info(f"Initialized Vertex AI Chat Model: {self.model_name}")

            self.tool_schemas_for_binding: List[Dict[str, Any]] = []
            self.vertex_tools: Optional[List[aiplatform.FunctionDeclaration]] = None

        except Exception as e:
            logger.exception(f"Failed to initialize Vertex AI: {e}")
            raise

    def _format_messages_for_vertex(self, messages: List[Any]) -> List[ChatMessage]:
        """Converts LangChain messages to Vertex AI's expected format."""
        vertex_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                vertex_messages.append(ChatMessage(content=msg.content, role="system")) # Include system messages
            elif isinstance(msg, HumanMessage):
                vertex_messages.append(ChatMessage(content=msg.content, role="user"))
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    function_calls = []
                    for tool_call in msg.tool_calls:
                        function_calls.append(
                            FunctionCall(
                                name=tool_call['name'],
                                arguments=json.dumps(tool_call['args'])
                            )
                        )
                    vertex_messages.append(ChatMessage(content=msg.content or "", role="assistant", function_call=function_calls[0] if function_calls else None)) # Vertex AI only supports one function call per message
                else:
                    vertex_messages.append(ChatMessage(content=msg.content, role="assistant"))
            elif isinstance(msg, ToolMessage):
                # Vertex AI expects tool responses as 'function' role
                vertex_messages.append(ChatMessage(content=msg.content, role="function"))
            else:
                logger.warning(f"Skipping unsupported message type: {type(msg)}")

        return vertex_messages

    def initializeTools(self, toolsList: list):
        """Stores tool schemas and creates the Vertex AI-formatted tool list."""
        if not toolsList:
            logger.warning("Received empty toolsList in initializeTools.")
            self.vertex_tools = None
            self.tool_schemas_for_binding = []
            return

        try:
            function_declarations = []
            self.tool_schemas_for_binding = toolsList

            for tool_schema in toolsList:
                updatedToolSchema = convert_json_a_to_b(tool_schema)

                name = updatedToolSchema.get("name")
                description = updatedToolSchema.get("description")
                parameters = updatedToolSchema.get("parameters")

                if not name or not description or not parameters:
                    logger.warning(f"Skipping tool due to missing name, description, or parameters: {tool_schema}")
                    continue

                declaration = aiplatform.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters
                )
                function_declarations.append(declaration)

            self.vertex_tools = function_declarations
            logger.info(f"Created Vertex AI Tool object with {len(function_declarations)} function declarations.")

        except Exception as e:
            logger.exception(f"Failed to create Vertex AI Tool object from schemas: {e}")
            self.vertex_tools = None


    async def invoke_structured_plan(self, messages: List[Any]) -> Plan:
        """Invokes the LLM requesting a structured output conforming to the Plan schema."""
        if not self.chat_model:
            raise RuntimeError("Vertex AI model not initialized.")

        last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_human_message:
            raise ValueError("Cannot generate plan without a user query (HumanMessage).")

        planning_contents = self._format_messages_for_vertex([last_human_message])

        try:
            logger.debug(f"Invoking Vertex AI for structured plan. Contents: {planning_contents}")

            response = self.chat_model.predict(
                planning_contents,
                tools=self.vertex_tools,
                temperature=0,
                response_schema=self.responseSchema
            )

            logger.info("Structured LLM invoked successfully for plan generation.")
            logger.debug(f"Raw Vertex Response: {response}")

            if not response.text:
                logger.error("No text content received from Vertex AI.")
                raise RuntimeError("LLM response invalid: No text content.")

            try:
                plan_data = json.loads(response.text)
                validated_plan = Plan(**plan_data)
                logger.debug(f"Received and validated Plan structure: {validated_plan.model_dump()}")
                return validated_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}. Raw text: {response.text}")
                raise
            except ValidationError as e:
                logger.error(f"Failed to validate Plan structure: {e}. Raw JSON: {response.text}")
                raise

        except Exception as e:
            logger.error(f"Error invoking structured LLM or processing response: {e}", exc_info=True)
            raise

    async def invoke_prompt(self, messages: List[Any]) -> aiplatform.preview.language_models.ChatMessage:
        """Invokes the LLM with the provided message history for a general response."""
        if not self.chat_model:
            raise RuntimeError("Vertex AI model not initialized.")

        vertex_formatted_messages = self._format_messages_for_vertex(messages)

        if not vertex_formatted_messages:
            logger.warning("invoke_prompt called with no user/model messages after formatting.")
            raise ValueError("No valid messages to send to the LLM.")

        try:
            logger.debug(f"Invoking Vertex AI for general prompt. Contents: {vertex_formatted_messages}")
            response = self.chat_model.predict(
                vertex_formatted_messages,
                tools=self.vertex_tools,
                temperature=0
            )
            logger.debug(f"Raw Vertex Response for invoke_prompt: {response}")
            return response

        except Exception as error:
            logger.error(f"Error occurred when processing prompt: {error}", exc_info=True)
            raise


# --- END OF FILE models.py ---
# --- START OF FILE models.py ---

import os
from pydantic import BaseModel, Field, model_validator, ValidationError
from ollama import chat
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
from icecream import ic
import uuid

# Get logger for this module
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables early for API key access

# --- Pydantic Schemas for Structured Output (Using Pydantic v2) ---

class OneActStep(BaseModel):
    prompt: str = Field(...)

# Pydantic v2 automatically handles forward references like 'ToolStep' in List['ToolStep']
# No need for ToolStep.update_forward_refs()

class Plan(BaseModel):
    """Defines the overall structure for the generated plan, which is a list of ToolSteps."""
    plan: List[OneActStep] = Field(..., description="An array of ToolSteps to execute in sequence to fulfill the user's request.")


class Models:
    def __init__(self):
        try:
            logger.info("Initializing Tools")
            self.tools = None
            self.model = "llama3.2:3b"
        except Exception as e:
            logger.exception(f"Failed to initialize: {e}")
            raise ValueError(f"Failed to initialize: {e}") from e

    def initializeTools(self, toolsList: list):
        """Binds tools to the LLM and then creates the structured LLM variant."""
        if not toolsList:
             logger.warning("Received empty toolsList in initializeTools. Structured output might not work as expected without function definitions.")
             # Decide how to handle this: maybe just use the base LLM or raise an error?
             # For now, let's proceed but log the warning.
             # self.structured_llm = self.llm.with_structured_output(Plan) # This might fail without tools depending on the model

        try:
            # Bind tools to the standard LLM instance first
            self.tools = toolsList
            logger.info(f"Bound {len(toolsList)} tools to the base Object instance.")

        except Exception as e:
            logger.exception(f"Failed to bind tools : {e}")
            raise

    async def invoke_structured_plan(self, messages) -> Plan:
        """
        Invokes the LLM requesting a structured output conforming to the Plan schema.
        """
        try:
            response = chat(
                messages = messages,
                model = self.model,
                format = Plan.model_json_schema(),
                tools = self.tools
            )
            try:
                content = ic(Plan.model_validate_json(response['message']['content']))
                logger.debug(f"Raw LLM response content: {content}")

                return content
            except ValidationError as ve:
                logger.error(f"Pydantic validation error for Plan: {ve}")
                raise ValueError(f"LLM response does not conform to Plan schema: {ve}") from ve

        except Exception as e:
            logger.error(f"Error invoking structured LLM or processing response: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate or validate structured plan: {e}")

    async def ainvoke(self, messages):
        try:
            response = chat(
                messages=messages,
                model=self.model,
                tools=self.tools
            )
            # Process tool_calls to normalize structure
            if 'message' in response and 'tool_calls' in response['message']:
                processed_tool_calls = []
                for tool_call in response['message']['tool_calls']:
                    tool_call_id = tool_call.get('id') or str(uuid.uuid4())
                    function = tool_call.get('function', {})  # Get the function dict
                    tool_call_dict = {
                        'id': tool_call_id,
                        'name': function.get('name'),  # Access name from function
                        'args': function.get('arguments', {})  # Access args from function
                    }
                    processed_tool_calls.append(tool_call_dict)
                response['message']['tool_calls'] = processed_tool_calls
            return response
        except Exception as error:
            logger.critical(f"Error occurred when processing prompt for a tool call: {error}")
            raise error

# --- END OF FILE models.py ---
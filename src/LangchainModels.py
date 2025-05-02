# --- START OF FILE models.py ---

import os
# Use Pydantic v2 imports
from pydantic import BaseModel, Field, model_validator, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import chat
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
# Removed ForwardRef and pydantic_v1 imports

# Get logger for this module
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables early for API key access

class ActionStep(BaseModel):
    prompt: str = Field(...)

# Pydantic v2 automatically handles forward references like 'ToolStep' in List['ToolStep']
# No need for ToolStep.update_forward_refs()

class Plan(BaseModel):
    """Defines the overall structure for the generated plan, which is a list of ToolSteps."""
    plan: List[ActionStep] = Field(..., description="An array of ToolSteps to execute in sequence to fulfill the user's request.")


class Models:
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-04-17",
                google_api_key=google_api_key,
            )
            logger.info("Initialized ChatGoogleGenerativeAI LLM.")
            self.llm_with_tools = None
            self.structured_llm = None
        except Exception as e:
            logger.exception(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise ValueError(f"Failed to initialize Google LLM: {e}") from e

    def initializeTools(self, toolsList: list):
        """Binds tools to the LLM and then creates the structured LLM variant."""
        if not toolsList:
             logger.warning("Received empty toolsList in initializeTools. Structured output might not work as expected without function definitions.")

        try:
            # Bind tools to the standard LLM instance first
            self.llm_with_tools = self.llm.bind_tools(toolsList)
            logger.info(f"Bound {len(toolsList)} tools to the base LLM instance.")

            # Create the structured LLM variant from the tool-bound LLM
            # self.structured_llm = self.llm.with_structured_output(Plan)
            self.structured_llm = self.llm.bind_tools(toolsList)
            self.structured_llm = self.structured_llm.with_structured_output(Plan)
            logger.info("Created structured LLM variant (Plan schema) from tool-bound LLM.")

        except Exception as e:
            logger.exception(f"Failed to bind tools or create structured LLM: {e}")
            # Specifically log if the schema itself might be the issue after Pydantic v2 migration
            if isinstance(e, (ValidationError, TypeError)): # Check for Pydantic or type errors during binding/structuring
                logger.error(f"Potential issue with Pydantic schema (Plan/ToolStep) interaction: {e}")
            raise

    async def invoke_structured_plan(self, messages) -> Plan:
        """
        Invokes the LLM requesting a structured output conforming to the Plan schema.
        """
        if not self.structured_llm:
            logger.error("Structured LLM is not initialized. Call initializeTools first.")
            raise RuntimeError("Structured LLM not initialized.")
        try:
            logger.debug("Invoking structured LLM (Plan schema, tools bound).")
            response = await self.structured_llm.ainvoke(messages)
            logger.info("Structured LLM invoked successfully for plan generation.")

            if not isinstance(response, Plan):
                from langchain_core.messages import AIMessage
                if isinstance(response, AIMessage):
                    logger.error("Received AIMessage instead of Plan object directly. Response content: %s", response.content)
                raise ValueError(f"LLM response was type {type(response)}, not the expected Plan structure.")

            logger.debug(f"Received Plan structure: {response.model_dump()}")
            return response
        except ValidationError as ve:
            logger.error(f"Pydantic validation error processing LLM response: {ve}", exc_info=True)
            raise RuntimeError(f"Failed to validate structured plan response: {ve}") from ve
        except Exception as e:
            logger.error(f"Error invoking structured LLM or processing response: {e}", exc_info=True)
            if "InvalidArgument" in str(e) and "token limit" in str(e).lower():
                logger.error("Token limit exceeded in Gemini API. Consider reducing MAX_TOKENS in .env or optimizing conversation history.")
                raise RuntimeError("Token limit exceeded. Please reduce context size or try a shorter query.") from e
            if "InvalidArgument" in str(e):
                logger.error("Google API InvalidArgument error, possibly due to schema or input size issues.")
            raise RuntimeError(f"Failed to generate or validate structured plan: {e}") from e

# --- END OF FILE models.py ---
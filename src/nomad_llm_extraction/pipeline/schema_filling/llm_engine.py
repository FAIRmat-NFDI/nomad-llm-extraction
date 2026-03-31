"""(Outlines/Instructor Constrained Decoding).

Example Usage:

    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import OutlinesEngine, InstructorEngine

    # For Local vLLM
    # active_engine = OutlinesEngine(model_name="Qwen/Qwen2.5-72B", api_url="http://localhost:8000/v1")

    # For Local Ollama
    # active_engine = InstructorEngine(model_name="llama3.1", api_url="http://localhost:11434/v1")

    # For Cloud ChatGPT
    # active_engine = InstructorEngine(model_name="gpt-4o")
"""
import logging
from typing import Type, TypeVar, Any, Optional
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class StructuredLLMEngine:
    """Base class for all structured extraction. its entire job is to define a strict contract or blueprint."""
    def generate(self, prompt: str, response_model: Type[T], temperature: float = 0.1) -> T:
        raise NotImplementedError("Subclasses must implement the generate method.")

class OutlinesEngine(StructuredLLMEngine):
    def __init__(self, model_name: str, api_url: Optional[str] = None, api_key: str = "EMPTY"):
        import outlines
        self.model_name = model_name
        
        if api_url:
            self.client = OpenAI(base_url=api_url, api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key if api_key != "EMPTY" else None)
            
        # Using user's original logic
        self.model = outlines.from_vllm(self.client, model_name)
        logger.info(f"Initialized OutlinesEngine for {model_name}")

    def generate(self, prompt: str, response_model: Type[T], temperature: float = 0.1) -> T:
        try:
            # tell outlines to use the passed pydantic model
            result_json = self.model(prompt, output_type=response_model, temperature=temperature)
            return response_model.model_validate_json(result_json)
        except Exception as e:
            logger.error(f"Outlines generation failed: {e}")
            raise

class InstructorEngine(StructuredLLMEngine):
    def __init__(self, model_name: str, api_url: Optional[str] = None, api_key: str = "EMPTY"):
        import instructor
        self.model_name = model_name
        
        if api_url:
            base_client = OpenAI(base_url=api_url, api_key=api_key)
        else:
            base_client = OpenAI(api_key=api_key if api_key != "EMPTY" else None)
            
        self.client = instructor.from_openai(base_client)
        logger.info(f"Initialized InstructorEngine for {model_name}")

    def generate(self, prompt: str, response_model: Type[T], temperature: float = 0.1) -> T:
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                response_model=response_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Instructor generation failed: {e}")
            raise


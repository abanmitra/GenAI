import requests
from langsmith import Client
from typing import List, Dict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIToolsIntegration:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", langsmith_api_key: str = None):
        """
        Initialize the AI Tools Integration class
        
        Args:
            ollama_base_url (str): Base URL for Ollama API
            langsmith_api_key (str): API key for Langsmith
        """
        self.ollama_base_url = ollama_base_url
        self.langsmith_client = Client(api_key=langsmith_api_key) if langsmith_api_key else None
        
    def list_ollama_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"Failed to get models. Status code: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def generate_with_ollama(self, model_name: str, prompt: str, params: Dict = None) -> str:
        """
        Generate text using an Ollama model
        
        Args:
            model_name (str): Name of the Ollama model to use
            prompt (str): Input prompt for generation
            params (Dict): Additional parameters for generation
        
        Returns:
            str: Generated text
        """
        try:
            default_params = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            if params:
                default_params.update(params)
                
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=default_params
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Generation failed. Status code: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return ""

    def track_with_langsmith(self, run_name: str, inputs: Dict, outputs: Dict) -> None:
        """
        Track runs with Langsmith
        
        Args:
            run_name (str): Name of the run to track
            inputs (Dict): Input data
            outputs (Dict): Output data
        """
        if not self.langsmith_client:
            logger.warning("Langsmith client not initialized. Skipping tracking.")
            return
            
        try:
            self.langsmith_client.create_run(
                name=run_name,
                inputs=inputs,
                outputs=outputs,
                run_type="llm"
            )
        except Exception as e:
            logger.error(f"Error tracking with Langsmith: {str(e)}")

def main():
    # Initialize the integration
    integration = AIToolsIntegration(
        langsmith_api_key="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"  # Replace with actual API key
    )
    
    # List available models
    available_models = integration.list_ollama_models()
    logger.info(f"Available models: {available_models}")
    
    # Example generation
    prompt = "Explain the concept of machine learning in simple terms."
    model_name = "llama3.2"  # Or any other available model
    
    generation_params = {
        "temperature": 0.8,
        "max_tokens": 300
    }
    
    # Generate response
    response = integration.generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        params=generation_params
    )
    
    # Track the generation with Langsmith
    integration.track_with_langsmith(
        run_name="concept_explanation",
        inputs={"prompt": prompt, "model": model_name},
        outputs={"response": response}
    )
    
    logger.info(f"Generated response: {response}")

if __name__ == "__main__":
    main()
"""
LLM Service: Handles Gemini API calls for product extraction
"""

import logging
import os
import json
from typing import Optional

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles LLM (Gemini) API calls for intelligent product extraction.
    """
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize LLM Service with Gemini.
        
        Args:
            project_id: Google Cloud Project ID
            credentials_path: Path to service account credentials
        """
        try:
            # Initialize Vertex AI
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            # Set credentials if path provided
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location="us-central1")
            
            # Initialize Gemini model
            self.model = GenerativeModel("gemini-2.0-flash")
            
            logger.info("✅ LLM Service (Gemini) initialized successfully")
        
        except ImportError:
            logger.error("❌ vertexai library not found. Install: pip install google-cloud-aiplatform")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
            raise
    
    def extract_products(self, prompt: str) -> str:
        """
        Call Gemini to extract product information.
        
        Args:
            prompt: Formatted prompt for product extraction
            
        Returns:
            JSON response from Gemini
        """
        try:
            logger.debug("Calling Gemini for product extraction...")
            
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            response_text = response.text
            
            logger.debug(f"Gemini response received: {response_text[:100]}...")
            
            return response_text
        
        except Exception as e:
            logger.error(f"❌ Gemini API error: {str(e)}")
            raise
    
    def classify_text(self, text: str, categories: list) -> dict:
        """
        Classify text into one of given categories.
        
        Args:
            text: Text to classify
            categories: List of categories
            
        Returns:
            Classification result
        """
        try:
            prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}

Text: "{text}"

Return ONLY valid JSON:
{{"category": "chosen_category", "confidence": 0.95}}"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse and return
            return json.loads(response_text)
        
        except Exception as e:
            logger.error(f"❌ Classification error: {str(e)}")
            raise       
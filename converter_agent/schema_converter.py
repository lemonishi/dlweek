"""Document to JSON Schema converter using OpenAI GPT."""

import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class SchemaConverter:
    """Convert documents to JSON Schema using OpenAI GPT."""
    
    SCHEMA_GENERATION_PROMPT = """
You are an expert in JSON schema design. Your task is to analyze a document and generate a comprehensive JSON schema that represents its structure.

Document Content:
{document_content}

Please generate a JSON schema that:
1. Captures all key entities and attributes from the document
2. Defines appropriate data types for each field
3. Includes proper validation rules
4. Marks required vs optional fields
5. Includes descriptions for each field

Return ONLY valid JSON schema without any additional text or markdown formatting.
"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 4096):
        """
        Initialize the schema converter.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"SchemaConverter initialized with model: {model}")
    
    def generate_schema_step_by_step(self, document_content: str, document_name: str = "Document") -> Dict[str, Any]:
        """
        Generate JSON schema from document content step by step.
        
        Args:
            document_content: The document content to analyze
            document_name: Name of the document (for logging)
            
        Returns:
            Generated JSON schema
            
        Raises:
            ValueError: If schema generation fails
        """
        logger.info(f"Step 1: Starting schema generation for '{document_name}'")
        logger.info(f"Step 2: Preparing prompt with document content (length: {len(document_content)})")
        
        # Step 1: Prepare the prompt
        prompt = self.SCHEMA_GENERATION_PROMPT.format(
            document_content=document_content[:3000]  # Limit content to 3000 chars for API
        )
        
        logger.info("Step 3: Calling OpenAI API for schema generation")
        
        # Step 2: Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert JSON schema designer. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            logger.info("Step 4: Received response from OpenAI API")
            
        except Exception as e:
            logger.error(f"Failed to call OpenAI API: {str(e)}")
            raise
        
        # Step 3: Parse the response
        logger.info("Step 5: Parsing and validating the schema response")
        
        schema_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if schema_text.startswith("```"):
            schema_text = "\n".join(schema_text.split("\n")[1:])
        if schema_text.endswith("```"):
            schema_text = "\n".join(schema_text.split("\n")[:-1])
        
        try:
            schema = json.loads(schema_text)
            logger.info("Step 6: Schema validation successful")
            return schema
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse schema JSON: {str(e)}")
            logger.error(f"Schema text: {schema_text}")
            raise ValueError(f"Invalid JSON schema generated: {str(e)}")
    
    def refine_schema(self, initial_schema: Dict[str, Any], requirements: str) -> Dict[str, Any]:
        """
        Refine an existing schema based on additional requirements.
        
        Args:
            initial_schema: The initial schema to refine
            requirements: Additional requirements or feedback
            
        Returns:
            Refined JSON schema
        """
        logger.info("Refining schema with additional requirements")
        
        refinement_prompt = f"""
You have the following JSON schema:

{json.dumps(initial_schema, indent=2)}

Please refine this schema based on these requirements:
{requirements}

Return the refined JSON schema in valid JSON format only.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert JSON schema designer. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": refinement_prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        schema_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks
        if schema_text.startswith("```"):
            schema_text = "\n".join(schema_text.split("\n")[1:])
        if schema_text.endswith("```"):
            schema_text = "\n".join(schema_text.split("\n")[:-1])
        
        return json.loads(schema_text)
    
    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate the generated schema structure.
        
        Args:
            schema: The schema to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = {'type', 'properties'}
        
        # Check if it has required keys or is an object type
        if '$schema' not in schema and 'type' in schema:
            return True
        
        if isinstance(schema, dict) and any(key in schema for key in required_keys):
            return True
        
        logger.warning("Schema validation warning: Missing expected keys")
        return False

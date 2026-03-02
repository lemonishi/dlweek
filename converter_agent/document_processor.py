"""Document processing utilities."""

import json
from pathlib import Path
from typing import Union, Dict, Any


class DocumentProcessor:
    """Handle document reading and processing."""
    
    SUPPORTED_FORMATS = {'.txt', '.json', '.yaml', '.yml', '.csv'}
    
    @staticmethod
    def read_document(file_path: Union[str, Path]) -> str:
        """
        Read a document from various formats.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if file_path.suffix.lower() not in DocumentProcessor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {DocumentProcessor.SUPPORTED_FORMATS}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    @staticmethod
    def load_json_document(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a JSON document.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON content
        """
        content = DocumentProcessor.read_document(file_path)
        return json.loads(content)
    
    @staticmethod
    def get_document_context(content: str, max_length: int = 1000) -> str:
        """
        Extract relevant context from document.
        
        Args:
            content: Document content
            max_length: Maximum length of context
            
        Returns:
            Document context or preview
        """
        lines = content.split('\n')
        context = '\n'.join(lines[:50])  # First 50 lines
        
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context
    
    @staticmethod
    def validate_content(content: str) -> bool:
        """
        Validate if content is suitable for schema generation.
        
        Args:
            content: Document content
            
        Returns:
            True if content is valid, False otherwise
        """
        return len(content.strip()) > 0

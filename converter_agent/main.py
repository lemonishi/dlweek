#!/usr/bin/env python3
"""Main entry point for document-to-JSON-schema converter."""

import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from config import Config
from document_processor import DocumentProcessor
from schema_converter import SchemaConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def convert_document_to_schema(
    document_path: str,
    output_path: Optional[str] = None,
    refine_requirements: Optional[str] = None,
    verbose: bool = False
) -> dict:
    """
    Main function to convert a document to JSON schema.
    
    Args:
        document_path: Path to the document file
        output_path: Optional path to save the schema
        refine_requirements: Optional refinement requirements
        verbose: Enable verbose logging
        
    Returns:
        Generated JSON schema
    """
    setup_logging(verbose)
    
    logger.info("=" * 60)
    logger.info("Document to JSON Schema Converter")
    logger.info("=" * 60)
    
    # Step 1: Validate configuration
    logger.info("\n📋 Step 1: Validating configuration...")
    try:
        Config.validate()
        logger.info("✓ Configuration validated successfully")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        raise
    
    # Step 2: Read the document
    logger.info("\n📄 Step 2: Reading document...")
    try:
        document_content = DocumentProcessor.read_document(document_path)
        logger.info(f"✓ Document read successfully (size: {len(document_content)} characters)")
    except Exception as e:
        logger.error(f"✗ Failed to read document: {e}")
        raise
    
    # Step 3: Validate document content
    logger.info("\n✔️  Step 3: Validating document content...")
    if not DocumentProcessor.validate_content(document_content):
        logger.error("✗ Document content is empty or invalid")
        raise ValueError("Invalid document content")
    logger.info("✓ Document content validated")
    
    # Step 4: Initialize schema converter
    logger.info("\n🤖 Step 4: Initializing OpenAI schema converter...")
    try:
        converter = SchemaConverter(
            api_key=Config.OPENAI_API_KEY,
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        logger.info(f"✓ Converter initialized (Model: {Config.MODEL_NAME})")
    except Exception as e:
        logger.error(f"✗ Failed to initialize converter: {e}")
        raise
    
    # Step 5: Generate schema
    logger.info("\n🔄 Step 5: Generating JSON schema using GPT...")
    try:
        doc_name = Path(document_path).name
        schema = converter.generate_schema_step_by_step(
            document_content=document_content,
            document_name=doc_name
        )
        logger.info("✓ Schema generated successfully")
    except Exception as e:
        logger.error(f"✗ Failed to generate schema: {e}")
        raise
    
    # Step 6: Optional schema refinement
    if refine_requirements:
        logger.info("\n✨ Step 6: Refining schema based on requirements...")
        try:
            schema = converter.refine_schema(schema, refine_requirements)
            logger.info("✓ Schema refined successfully")
        except Exception as e:
            logger.error(f"✗ Failed to refine schema: {e}")
            raise
    
    # Step 7: Validate the generated schema
    logger.info("\n🔍 Step 7: Validating generated schema...")
    if converter.validate_schema(schema):
        logger.info("✓ Schema validation passed")
    else:
        logger.warning("⚠️  Schema validation warning (but continuing)")
    
    # Step 8: Save the schema
    logger.info("\n💾 Step 8: Saving schema to file...")
    try:
        if output_path is None:
            doc_path = Path(document_path)
            output_path = Config.OUTPUT_PATH / f"{doc_path.stem}_schema.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"✓ Schema saved to: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save schema: {e}")
        raise
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Conversion completed successfully!")
    logger.info("=" * 60 + "\n")
    
    return schema


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert documents to JSON Schema using OpenAI GPT"
    )
    parser.add_argument(
        "document",
        help="Path to the document file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for the JSON schema"
    )
    parser.add_argument(
        "-r", "--refine",
        help="Additional refinement requirements for the schema"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        schema = convert_document_to_schema(
            document_path=args.document,
            output_path=args.output,
            refine_requirements=args.refine,
            verbose=args.verbose
        )
        print("\nGenerated Schema:")
        print(json.dumps(schema, indent=2))
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        exit(1)


if __name__ == "__main__":
    main()

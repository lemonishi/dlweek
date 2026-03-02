#!/usr/bin/env python3
"""Example usage of the document-to-JSON-schema converter."""

import json
from pathlib import Path
from schema_converter import SchemaConverter
from config import Config

def example_basic_conversion():
    """Example: Basic document to schema conversion."""
    print("Example 1: Basic Document to Schema Conversion\n")
    
    # Sample document content
    sample_document = """
    Customer Information:
    - Name: John Doe
    - Email: john@example.com
    - Phone: +1-234-567-8900
    - Address: 123 Main St, City, State 12345
    - Account Status: Active
    - Created Date: 2024-01-15
    - Credit Limit: $5000
    """
    
    # Initialize converter
    converter = SchemaConverter(
        api_key=Config.OPENAI_API_KEY,
        model=Config.MODEL_NAME
    )
    
    # Generate schema
    try:
        schema = converter.generate_schema_step_by_step(
            document_content=sample_document,
            document_name="customer_info"
        )
        
        print("Generated Schema:")
        print(json.dumps(schema, indent=2))
        
        # Save to file
        output_file = Config.OUTPUT_PATH / "customer_schema.json"
        with open(output_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        print(f"\nSchema saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_with_refinement():
    """Example: Schema generation with refinement."""
    print("\n\nExample 2: Schema Generation with Refinement\n")
    
    # Sample document
    sample_document = """
    Product Details:
    Product ID: P-001
    Name: Wireless Headphones
    Brand: AudioTech
    Price: $79.99
    Stock: 150 units
    Warranty: 2 years
    Colors Available: Black, White, Blue
    Features: Noise cancellation, 20h battery
    """
    
    # Initialize converter
    converter = SchemaConverter(
        api_key=Config.OPENAI_API_KEY,
        model=Config.MODEL_NAME
    )
    
    try:
        # Generate initial schema
        schema = converter.generate_schema_step_by_step(
            document_content=sample_document,
            document_name="product_info"
        )
        
        print("Initial Schema:")
        print(json.dumps(schema, indent=2))
        
        # Refine with specific requirements
        refinement_requirements = """
        - Add sku field as string (required)
        - Make colors array with minimum 1 item
        - Add warranty period in months as integer
        - Add deprecation notice field
        - Group color options separately from other attributes
        """
        
        refined_schema = converter.refine_schema(schema, refinement_requirements)
        
        print("\n\nRefined Schema:")
        print(json.dumps(refined_schema, indent=2))
        
        # Save refined schema
        output_file = Config.OUTPUT_PATH / "product_schema_refined.json"
        with open(output_file, 'w') as f:
            json.dump(refined_schema, f, indent=2)
        
        print(f"\nRefined schema saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Document to JSON Schema Converter - Examples\n")
    print("=" * 60)
    
    try:
        Config.validate()
        example_basic_conversion()
        example_with_refinement()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set up your .env file with the required API key.")

#!/usr/bin/env python3
"""
Multi-Dataset Polynomial Fitting & Analysis Tool - Main Application

This is the main entry point for the modular polynomial fitting application.
The application has been refactored from a single large file into multiple modules
for better maintainability and organization.

Modules:
- src.data_input: Data parsing and file handling
- src.math_functions: Mathematical operations and fitting models
- src.statistics: Statistical analysis and formatting
- src.dataset_management: Multi-dataset configuration management
- src.plotting: Plotting and visualization functions
- src.ui_components: Gradio UI layout and components
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the UI creation function
from src.ui_components import create_gradio_interface

def main():
    """Main function to launch the application."""
    print("🚀 Starting Multi-Dataset Polynomial Fitting & Analysis Tool...")
    print("📁 Loading modular components...")
    
    try:
        # Create the Gradio interface
        demo = create_gradio_interface()
        
        print("✅ Application loaded successfully!")
        print("🌐 Launching web interface...")
        
        # Launch the application
        demo.launch(
            share=False,  # Set to True if you want a public link
            server_name="0.0.0.0",  # Local access only
            server_port=3000,  # Default Gradio port
            show_error=True,  # Show error messages in the interface
            quiet=False  # Show startup messages
        )
        
    except Exception as e:
        print(f"❌ Error launching application: {str(e)}")
        print("💡 Please check that all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()

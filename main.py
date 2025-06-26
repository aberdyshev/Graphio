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
import socket
from contextlib import closing
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the UI creation function
from src.ui_components import create_gradio_interface
                                                                                                        # я выбираю какой-то порт 
def find_free_port():
    """Находит полностью свободный порт."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # 0 = OS выберет свободный порт
        s.listen(1)
        return s.getsockname()[1]  # Возвращает номер порта

def main():
    """Main function to launch the application."""
    print("🚀 Starting Multi-Dataset Polynomial Fitting & Analysis Tool...")
    print("📁 Loading modular components...")
    
    try:
        # Create the Gradio interface
        demo = create_gradio_interface()                                                                 # интерфейс -> ui_components
        
        print("✅ Application loaded successfully!")
        print("🌐 Launching web interface...")
        
        # Launch the application


        demo.launch(
            share=False,  # Set to True if you want a public link
            server_name="127.0.0.1",  # Local access only
            server_port= find_free_port(),  # Default Gradio port               # запуская на каком-то iD
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

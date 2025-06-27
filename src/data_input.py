"""
Data Input and Parsing Module

This module handles all data input, parsing, and validation operations.
"""

import numpy as np
import pandas as pd
import re
from typing import Tuple, Optional, Union


def parse_input_data(text_data: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Parses comma, space, or newline separated numbers into a numpy array."""
    if not text_data.strip():
        return np.array([]), None
        
    normalized_text = re.sub(r',(\d)', r'.\1', text_data.strip())Add commentMore actions

    # Replace commas and newlines with spaces, then split
    numbers_str = re.split(r'[,\s\n]+', normalized_text)

    try:
        # Filter out empty strings resulting from multiple delimiters
        numbers = []
        invalid_entries = []
        
        for i, n in enumerate(numbers_str):
            if n:  # Skip empty strings
                try:
                    numbers.append(float(n))
                except ValueError:
                    invalid_entries.append(f"'{n}' at position {i+1}")
        
        if invalid_entries:
            error_msg = f"❌ Invalid number(s) found: {', '.join(invalid_entries[:3])}"
            if len(invalid_entries) > 3:
                error_msg += f" and {len(invalid_entries)-3} more"
            return None, error_msg
        
        if not numbers:
            return np.array([]), "⚠️ No valid numbers found in input"
            
        return np.array(numbers), None
        
    except Exception as e:
        return None, f"❌ Unexpected error parsing input: {str(e)}"


def handle_file_change(uploaded_file):
    """Handle file upload and extract column names for selection."""
    import gradio as gr
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file.name)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file.name)
            else:
                return (
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    None,
                    gr.Textbox(value=""),
                    gr.Textbox(value=""),
                    "❌ Unsupported file format. Please use CSV or Excel files."
                )
            
            # Get numeric columns
            numeric_columns = []
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce').dropna()
                    numeric_columns.append(col)
                except:
                    continue
            
            if not numeric_columns:
                return (
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    None,
                    gr.Textbox(value=""),
                    gr.Textbox(value=""),
                    "❌ No numeric columns found in the file."
                )
            
            # Set default selections
            x_default = numeric_columns[0] if len(numeric_columns) >= 1 else None
            y_default = numeric_columns[1] if len(numeric_columns) >= 2 else numeric_columns[0]
            z_default = numeric_columns[2] if len(numeric_columns) >= 3 else None
            
            success_msg = f"✅ File loaded successfully! Found {len(df)} rows and {len(numeric_columns)} numeric columns."
            
            return (
                gr.Dropdown(choices=numeric_columns, value=x_default, label="X Column"),
                gr.Dropdown(choices=numeric_columns, value=y_default, label="Y Column"),
                gr.Dropdown(choices=numeric_columns + [None], value=z_default, label="Z Column (Optional)"),
                df,
                gr.Textbox(value=""),
                gr.Textbox(value=""),
                success_msg
            )
            
        except Exception as e:
            return (
                gr.Dropdown(choices=[], value=None),
                gr.Dropdown(choices=[], value=None),
                gr.Dropdown(choices=[], value=None),
                None,
                gr.Textbox(value=""), 
                gr.Textbox(value=""),
                f"❌ Error processing file: {str(e)}"
            )
    else:
        # File cleared
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None), 
            gr.Dropdown(choices=[], value=None),
            None,  # Clear DataFrame
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            "ℹ️ File input cleared. Manual text input is now active."
        )

# --- START OF FILE dataset_management.py ---

"""
Dataset Management Module

This module handles multiple dataset configuration, management, and state operations.
"""

import gradio as gr
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import tempfile # For temporary file saving
import os
from datetime import datetime # For unique filenames


def create_new_curve(curve_configs: List[Dict], curve_names: List[str], current_curve_idx: int) -> Tuple[List[Dict], List[str], int, gr.Dropdown, str]:
    """Create a new curve configuration."""
    new_idx = len(curve_configs)
    new_name = f"Dataset {new_idx + 1}"

    # Default configuration
    default_config = {
        'name': new_name,
        'x_text': '',
        'y_text': '',
        'z_text': '',
        'degree': 3,
        'data_color': f'#{hash(new_name) % 0xFFFFFF:06x}',
        'fit_color': f'#{(hash(new_name) + 123456) % 0xFFFFFF:06x}',
        'data_marker': 'circle',
        'fit_line_style': 'solid',
        'fit_type': 'polynomial',
        'show_fit': True,
        'show_data': True,
        'x_errors_text': '',
        'y_errors_text': '',
        'show_error_bars': False,
        'file_df': None,
        'x_col_name': None,
        'y_col_name': None,
        'z_col_name': None,
        'visible': True,
        'force_3d': False
    }

    curve_configs.append(default_config)
    curve_names.append(new_name)

    return (
        curve_configs,
        curve_names,
        new_idx,
        gr.Dropdown(choices=curve_names, value=new_name, label="Select Dataset"),
        f"✅ Created new dataset: {new_name}"
    )


def duplicate_curve(curve_configs: List[Dict], curve_names: List[str], current_curve_idx: int) -> Tuple[List[Dict], List[str], int, gr.Dropdown, str]:
    """Duplicate the current curve configuration."""
    if not curve_configs or current_curve_idx >= len(curve_configs):
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names, value=curve_names[0] if curve_names else None), "❌ No dataset to duplicate"

    current_config = curve_configs[current_curve_idx].copy()
    new_idx = len(curve_configs)
    new_name = f"{current_config.get('name', 'Dataset')} (Copy)" # Use .get for safety
    current_config['name'] = new_name
    # Generate new distinct colors for the duplicate
    current_config['data_color'] = f'#{hash(new_name) % 0xFFFFFF:06x}'
    current_config['fit_color'] = f'#{(hash(new_name) * 7 + 98765) % 0xFFFFFF:06x}'


    curve_configs.append(current_config)
    curve_names.append(new_name)

    return (
        curve_configs,
        curve_names,
        new_idx,
        gr.Dropdown(choices=curve_names, value=new_name, label="Select Dataset"),
        f"✅ Duplicated dataset: {new_name}"
    )


def delete_curve(curve_configs: List[Dict], curve_names: List[str], current_curve_idx: int) -> Tuple[List[Dict], List[str], int, gr.Dropdown, str]:
    """Delete the current curve."""
    if len(curve_configs) <= 1:
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names, value=curve_names[0] if curve_names else None), "❌ Cannot delete the last dataset"

    if current_curve_idx >= len(curve_configs):
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names, value=curve_names[0] if curve_names else None), "❌ Invalid dataset index"

    deleted_name = curve_names[current_curve_idx]
    curve_configs.pop(current_curve_idx)
    curve_names.pop(current_curve_idx)

    # Adjust current index
    new_idx = min(current_curve_idx, len(curve_configs) - 1)

    return (
        curve_configs,
        curve_names,
        new_idx,
        gr.Dropdown(choices=curve_names, value=curve_names[new_idx], label="Select Dataset"),
        f"✅ Deleted dataset: {deleted_name}"
    )


# Update switch_curve to return the dataset name as the first UI output
def switch_curve(curve_configs: List[Dict], curve_names: List[str], selected_curve_name: str) -> Tuple:
    """Switch to a different curve configuration."""
    # The first output is current_curve_idx_state
    # The subsequent outputs update the UI components for the selected curve
    # The last output is the status message

    if selected_curve_name not in curve_names:
        # Return default updates for all UI components + status message
        # Number of UI components returned: dataset_name_input + x_input + y_input + z_input + degree_slider + data_color_picker + fit_color_picker + data_marker_dropdown + fit_line_dropdown + fit_type_dropdown + show_fit_checkbox + show_data_checkbox + x_errors_input + y_errors_input + show_error_bars_checkbox + visible_checkbox + force_3d_checkbox = 17
        # Total outputs = 1 (index) + 17 (UI components) + 1 (status) = 19
        return tuple([0] + [gr.update() for _ in range(17)] + ["❌ Selected dataset not found"])

    curve_idx = curve_names.index(selected_curve_name)
    config = curve_configs[curve_idx]

    # Return all the UI component updates for the selected curve
    return (
        curve_idx,  # current_curve_idx_state
        gr.Textbox(value=config.get('name', '')), # Added dataset_name_input value
        gr.Textbox(value=config.get('x_text', '')),  # x_input
        gr.Textbox(value=config.get('y_text', '')),  # y_input
        gr.Textbox(value=config.get('z_text', '')),  # z_input
        gr.Slider(value=config.get('degree', 3)),  # degree_slider
        gr.ColorPicker(value=config.get('data_color', '#1f77b4')),  # data_color_picker
        gr.ColorPicker(value=config.get('fit_color', '#ff7f0e')),  # fit_color_picker
        gr.Dropdown(value=config.get('data_marker', 'circle')),  # data_marker_dropdown
        gr.Dropdown(value=config.get('fit_line_style', 'solid')),  # fit_line_dropdown
        gr.Dropdown(value=config.get('fit_type', 'polynomial')),  # fit_type_dropdown
        gr.Checkbox(value=config.get('show_fit', True)),  # show_fit_checkbox
        gr.Checkbox(value=config.get('show_data', True)),  # show_data_checkbox
        gr.Textbox(value=config.get('x_errors_text', '')),  # x_errors_input
        gr.Textbox(value=config.get('y_errors_text', '')),  # y_errors_input
        gr.Checkbox(value=config.get('show_error_bars', False)),  # show_error_bars_checkbox
        gr.Checkbox(value=config.get('visible', True)),  # visible_checkbox
        gr.Checkbox(value=config.get('force_3d', False)), # force_3d_checkbox
        f"✅ Switched to dataset: {selected_curve_name}"  # curve_status
    )


# Update save_current_curve_config to accept the dataset name
def save_current_curve_config(
    curve_configs: List[Dict], current_curve_idx: int,
    dataset_name: str, # Added dataset_name parameter
    x_text: str, y_text: str, z_text: str, degree: int, data_color: str, fit_color: str,
    data_marker: str, fit_line_style: str, fit_type: str, show_fit: bool, show_data: bool,
    x_errors_text: str, y_errors_text: str, show_error_bars: bool, visible: bool, force_3d: bool,
    file_df: pd.DataFrame, x_col_name: str, y_col_name: str, z_col_name: str
) -> List[Dict]: # This function only needs to return the updated curve_configs state
    """Save the current UI state to the curve configuration."""
    if not curve_configs or current_curve_idx >= len(curve_configs):
        # This happens if the last dataset was deleted, but an event was pending.
        # Or if somehow current_curve_idx is out of sync.
        print(f"Warning: Attempted to save config for index {current_curve_idx}, but curve_configs has {len(curve_configs)} items.")
        return curve_configs # Return the original configs unchanged

    config = curve_configs[current_curve_idx]

    # Map input parameters to configuration keys
    # Ensure the order matches the function signature
    config['name'] = dataset_name # Save the name
    config['x_text'] = x_text
    config['y_text'] = y_text
    config['z_text'] = z_text
    config['degree'] = degree
    config['data_color'] = data_color
    config['fit_color'] = fit_color
    config['data_marker'] = data_marker
    config['fit_line_style'] = fit_line_style
    config['fit_type'] = fit_type
    config['show_fit'] = show_fit
    config['show_data'] = show_data
    config['x_errors_text'] = x_errors_text
    config['y_errors_text'] = y_errors_text
    config['show_error_bars'] = show_error_bars
    config['visible'] = visible
    config['force_3d'] = force_3d
    config['file_df'] = file_df
    config['x_col_name'] = x_col_name
    config['y_col_name'] = y_col_name
    config['z_col_name'] = z_col_name

    # Note: Renaming a dataset here updates the config dictionary,
    # but the curve_names_state and the curve_selector dropdown choices
    # are only updated when create/delete/duplicate/load functions run.
    # The curve_selector *value* will show the new name because its value
    # is tied to config.get('name'), but the list of choices won't update
    # until the next event that explicitly rebuilds the dropdown choices.
    # This is a minor UI inconsistency but simpler than rebuilding the dropdown
    # on every config change.

    return curve_configs

# --- Configuration Management Functions ---
# (Keep these as they are)

def _prepare_curve_config_for_saving(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares a single curve's configuration for JSON serialization."""
    saved_config = config.copy() # Work on a copy
    if 'file_df' in saved_config and isinstance(saved_config['file_df'], pd.DataFrame):
        # Serialize DataFrame to JSON string using orient='split'
        try:
            saved_config['file_df'] = saved_config['file_df'].to_json(orient='split', indent=2)
        except Exception as e:
            print(f"Could not serialize DataFrame for curve {saved_config.get('name', 'Unnamed')}: {e}")
            saved_config['file_df'] = None
    elif 'file_df' in saved_config and saved_config['file_df'] is not None and not isinstance(saved_config['file_df'], (pd.DataFrame, str)):
        print(f"Warning: file_df for curve {saved_config.get('name', 'Unnamed')} is an unexpected type: {type(saved_config['file_df'])}. Clearing for save.")
        saved_config['file_df'] = None

    # Ensure other potentially non-serializable types are handled if they exist in the future.
    return saved_config

def _reconstruct_curve_config_after_loading(config_from_json: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstructs a single curve's configuration after loading from JSON."""
    loaded_config = config_from_json.copy() # Work on a copy
    if 'file_df' in loaded_config and isinstance(loaded_config['file_df'], str):
        # Attempt to deserialize JSON string back to DataFrame
        try:
            # Check if the string looks like split orient JSON
            if loaded_config['file_df'].strip().startswith('{'):
                 loaded_config['file_df'] = pd.read_json(loaded_config['file_df'], orient='split')
            else:
                 print(f"Warning: file_df string for curve {loaded_config.get('name', 'Unnamed')} doesn't look like 'split' JSON. Setting to None.")
                 loaded_config['file_df'] = None

        except Exception as e:
            print(f"Could not deserialize file_df for curve {loaded_config.get('name', 'Unnamed')}: {e}")
            loaded_config['file_df'] = None # Fallback to None if deserialization fails

    elif 'file_df' in loaded_config and loaded_config['file_df'] is None:
        pass # It's already None, which is fine
    elif 'file_df' in loaded_config and isinstance(loaded_config['file_df'], list): # Common if to_dict('records') was used or incorrect serialization
        try:
            loaded_config['file_df'] = pd.DataFrame.from_records(loaded_config['file_df'])
        except Exception as e:
            print(f"Could not deserialize file_df from list of records for curve {loaded_config.get('name', 'Unnamed')}: {e}")
            loaded_config['file_df'] = None
    elif 'file_df' in loaded_config:
        print(f"Warning: file_df for curve {loaded_config.get('name', 'Unnamed')} is an unexpected type after JSON load: {type(loaded_config['file_df'])}. Setting to None.")
        loaded_config['file_df'] = None

    # Ensure all expected keys exist, possibly with default values if missing in JSON
    default_config_structure = {
        'name': 'Unnamed Dataset', 'x_text': '', 'y_text': '', 'z_text': '',
        'degree': 3, 'data_color': '#1f77b4', 'fit_color': '#ff7f0e',
        'data_marker': 'circle', 'fit_line_style': 'solid', 'fit_type': 'polynomial',
        'show_fit': True, 'show_data': True, 'x_errors_text': '', 'y_errors_text': '',
        'show_error_bars': False, 'file_df': None, 'x_col_name': None,
        'y_col_name': None, 'z_col_name': None, 'visible': True, 'force_3d': False
    }
    for key, default_value in default_config_structure.items():
         if key not in loaded_config:
             loaded_config[key] = default_value

    return loaded_config

def save_app_configuration(all_curve_configs: List[Dict[str, Any]], base_filename: str = "plot_config.json") -> Tuple[Optional[str], Optional[str]]:
    """
    Saves the list of curve configurations to a JSON file in a temporary directory.
    Returns the path to the saved file and an error message (if any).
    """
    try:
        # Ensure each config is a standard dict before processing
        standardized_configs = [dict(config) for config in all_curve_configs]
        prepared_configs = [_prepare_curve_config_for_saving(config) for config in standardized_configs]

        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_filename = f"{os.path.splitext(base_filename)[0]}_{timestamp}{os.path.splitext(base_filename)[1]}"
        filepath = os.path.join(temp_dir, unique_filename)

        with open(filepath, 'w') as f:
            json.dump(prepared_configs, f, indent=2)
        return filepath, None
    except Exception as e:
        return None, f"Error saving configuration: {str(e)}"

def load_app_configuration(filepath: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Loads curve configurations from a JSON file.
    Returns the list of reconstructed configurations and an error message (if any).
    """
    if not filepath:
        return None, "No file path provided for loading."
    try:
        with open(filepath, 'r') as f:
            configs_from_json = json.load(f)

        if not isinstance(configs_from_json, list):
            return None, "Configuration file is not in the expected format (must be a list of dataset configurations)."

        reconstructed_configs = []
        for i, config_json in enumerate(configs_from_json):
            if not isinstance(config_json, dict):
                print(f"Warning: Item at index {i} in loaded configuration is not a dictionary. Skipping.")
                continue
            reconstructed_configs.append(_reconstruct_curve_config_after_loading(config_json))

        if not reconstructed_configs:
             return [], "Loaded configuration was empty or contained no valid datasets." # Return empty list if none were valid

        return reconstructed_configs, None
    except FileNotFoundError:
        return None, f"Configuration file not found: {filepath}"
    except json.JSONDecodeError:
        return None, f"Error decoding JSON from configuration file. Ensure it is a valid JSON: {filepath}"
    except Exception as e:
        return None, f"An unexpected error occurred while loading the configuration: {str(e)}"
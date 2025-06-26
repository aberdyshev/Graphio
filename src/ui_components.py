"""
Gradio User Interface Module

This module contains all the Gradio UI components and layout definitions.
"""

import gradio as gr
from .data_input import handle_file_change
from .dataset_management import (
    create_new_curve, duplicate_curve, delete_curve, switch_curve, save_current_curve_config,
    save_app_configuration, load_app_configuration # Added new functions
)
from .plotting import update_combined_plot, update_slider_only


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📊 Multi-Dataset Polynomial Fitting & Analysis Tool\n"
            "Upload files or input data manually to fit polynomials, analyze statistics, and visualize results. "
            "Each dataset can have its own polynomial fit, styling, and configuration. "
            "💡 **Z-axis support**: Add optional Z values for color mapping in both manual input and file upload."
        )

        # Dataset management state
        curve_configs_state = gr.State([{
            'name': 'Dataset 1',
            'x_text': '0, 1, 2, 3, 4, 5, 6',
            'y_text': '-1.2, 0, 0.6, 1.2, 2.4, 5.0, 9.8',
            'z_text': '',  # Add Z text field
            'degree': 3,
            'data_color': '#1f77b4',
            'fit_color': '#ff7f0e',
            'data_marker': 'circle',
            'fit_line_style': 'solid',
            'fit_type': 'polynomial',
            'show_fit': True,
            'show_data': True,
            'x_errors_text': '',
            'y_errors_text': '',
            'show_error_bars': False,
            'file_df': None, # This will be serialized/deserialized
            'x_col_name': None,
            'y_col_name': None,
            'z_col_name': None,
            'visible': True,
            'force_3d': False
        }])
        curve_names_state = gr.State(['Dataset 1'])
        current_curve_idx_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                # Configuration Management Section
                

                gr.Markdown("### 📊 Dataset Management")
                with gr.Row():
                    curve_selector = gr.Radio(
                        choices=['Dataset 1'],
                        value='Dataset 1',
                        label="Select Dataset",
                        interactive=True
                    )
                
                with gr.Row():
                    new_curve_btn = gr.Button("➕ Add Dataset", size="sm")
                    duplicate_curve_btn = gr.Button("📋 Duplicate", size="sm")
                    delete_curve_btn = gr.Button("🗑️ Delete", size="sm")
                
                curve_status = gr.Textbox(
                    label="Dataset Status",
                    interactive=False,
                    lines=1,
                    value="Dataset 1 active"
                )

                gr.Markdown("### 📥 Data Input")
                with gr.Tabs():
                    with gr.TabItem("📝 Manual Input"):
                        x_input = gr.Textbox(
                            label="X values",
                            placeholder="e.g., 1, 2, 3, 4, 5",
                            lines=3,
                            value="0, 1, 2, 3, 4, 5, 6"
                        )
                        y_input = gr.Textbox(
                            label="Y values",
                            placeholder="e.g., 0.5, 2.1, 3.8, 8.2, 12.5",
                            lines=3,
                            value="-1.2, 0, 0.6, 1.2, 2.4, 5.0, 9.8"
                        )
                        z_input = gr.Textbox(
                            label="Z values (Optional)",
                            placeholder="e.g., 10, 20, 15, 25, 30, 18, 22 (for color mapping)",
                            lines=2,
                            value=""
                        )
                    
                    with gr.TabItem("📁 File Upload"):
                        file_input = gr.File(
                            label="Upload CSV/Excel File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath"
                        )
                        
                        file_dataframe_state = gr.State(None)
                        
                        with gr.Row():
                            x_column_dropdown = gr.Dropdown(
                                choices=[], value=None, label="X Column", interactive=True
                            )
                            y_column_dropdown = gr.Dropdown(
                                choices=[], value=None, label="Y Column", interactive=True
                            )
                            z_column_dropdown = gr.Dropdown(
                                choices=[], value=None, label="Z Column (Optional)", interactive=True
                            )
                        
                        file_status_output = gr.Textbox(
                            label="File Status", 
                            interactive=False, 
                            lines=2,
                            value="No file uploaded"
                        )

                gr.Markdown("### ⚙️ Dataset Configuration")
                with gr.Row():
                    show_data_checkbox = gr.Checkbox(label="Show Data Points", value=True)
                    show_fit_checkbox = gr.Checkbox(label="Show Polynomial Fit", value=True)
                
                visible_checkbox = gr.Checkbox(label="Visible on Plot", value=True)
                
                degree_slider = gr.Slider(
                    minimum=0, maximum=10, step=1, value=3,
                    label="Polynomial Degree"
                )
                
                fit_type_dropdown = gr.Dropdown(
                    choices=["polynomial"],
                    value="polynomial",
                    label="Fitting Type"
                )
                
                force_3d_checkbox = gr.Checkbox(label="Attempt 3D Surface Plot (if Z data present)", value=False, interactive=True) # Added 3D checkbox

                with gr.Row():
                    data_color_picker = gr.ColorPicker(label="Data Color", value="#1f77b4")
                    fit_color_picker = gr.ColorPicker(label="Fit Color", value="#ff7f0e")
                
                with gr.Row():
                    data_marker_dropdown = gr.Dropdown(
                        choices=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star'],
                        value='circle', label="Data Marker"
                    )
                    fit_line_dropdown = gr.Dropdown(
                        choices=['solid', 'dot', 'dash', 'longdash', 'dashdot'],
                        value='solid', label="Fit Line Style"
                    )
                
                gr.Markdown("### 💾 Configuration Management")
                with gr.Row():
                    save_config_button = gr.Button("Save Configuration", size="sm")
                    load_config_file_input = gr.File(
                        label="Load Configuration File", 
                        file_types=[".json"], 
                        type="filepath",
                        scale=2 # Give more space to file input
                    )
                config_management_status = gr.Textbox(label="Config Status", interactive=False, lines=2, visible=False)
                download_config_link = gr.File(label="Download Saved Configuration", interactive=False, visible=False)

                # Error bars
                show_error_bars_checkbox = gr.Checkbox(label="Show Error Bars", value=False)
                x_errors_input = gr.Textbox(label="X Uncertainties", placeholder="Enter the error for each data point", lines=1)
                y_errors_input = gr.Textbox(label="Y Uncertainties", placeholder="enter the error for each data point", lines=1)

            with gr.Column(scale=2):
                gr.Markdown("### 📈 Multi-Dataset Visualization")
                plot_output = gr.Plot(label="Combined Plot", show_label=True)
                
                equation_output = gr.Textbox(
                    label="Plot Status", 
                    interactive=False, 
                    lines=2
                )
            
                # Main update function
                update_button = gr.Button("🚀 Update Multi-Dataset Plot", variant="primary", size="lg")
                


                with gr.Tabs():
                    with gr.TabItem("📊 Statistics"):
                        statistics_output = gr.Textbox(
                            label="Statistical Analysis", 
                            interactive=False, 
                            lines=15,
                            show_label=False
                        )
                    
                    with gr.TabItem("📐 Area Analysis"):
                        area_output = gr.Textbox(
                            label="Area Under Curve", 
                            interactive=False, 
                            lines=15,
                            show_label=False
                        )
                    
                    with gr.TabItem("🔮 Extrapolation"):
                        extrapolation_output = gr.Textbox(
                            label="Prediction Results", 
                            interactive=False, 
                            lines=12,
                            show_label=False
                        )

        with gr.Accordion("Plot Customization and Analysis Tools", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Display Options**")
                    plot_options_checkbox = gr.CheckboxGroup(
                        ["Show Grid", "Show Axes Labels", "Show Title"],
                        label="Plot Options",
                        value=["Show Grid", "Show Axes Labels", "Show Title"]
                    )
                    xaxis_scale_dropdown = gr.Dropdown(choices=["linear", "log"], value="linear", label="X-axis Scale")
                    yaxis_scale_dropdown = gr.Dropdown(choices=["linear", "log"], value="linear", label="Y-axis Scale")
                
                with gr.Column():
                    gr.Markdown("**Labels & Title**")
                    custom_x_label_input = gr.Textbox(label="X-axis Label", placeholder="X values")
                    custom_y_label_input = gr.Textbox(label="Y-axis Label", placeholder="Y values")
                    custom_title_input = gr.Textbox(label="Plot Title", placeholder="Multi-Dataset Analysis")
                
                with gr.Column():
                    gr.Markdown("**Font Settings**")
                    font_family_dropdown = gr.Dropdown(
                        choices=["Arial", "Times New Roman", "Courier New", "Georgia"],
                        value="Arial", label="Font Family"
                    )
                    title_font_size_slider = gr.Slider(8, 32, 16, label="Title Font Size")
                    axes_font_size_slider = gr.Slider(8, 24, 12, label="Axes Font Size")
                    legend_font_size_slider = gr.Slider(8, 20, 10, label="Legend Font Size")
                    font_color_picker = gr.ColorPicker(label="Font Color", value="#000000")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Special Points**")
                    special_points_checkbox = gr.CheckboxGroup(
                        ["Show Extrema", "Show X-Intercepts", "Show Y-Intercept"],
                        label="Special Points", value=[]
                    )
                    x_derivative_input = gr.Number(label="X for f'(x)", info="Optional")
                    derivative_output_text = gr.Textbox(label="Derivative", interactive=False, lines=1)
                
                with gr.Column():
                    gr.Markdown("**Area Under Curve**")
                    show_area_checkbox = gr.Checkbox(label="Show Area", value=False)
                    area_start_x_input = gr.Number(label="Start X")
                    area_end_x_input = gr.Number(label="End X")
                
                with gr.Column():
                    gr.Markdown("**Extrapolation**")
                    show_extrapolation_checkbox = gr.Checkbox(label="Show Extrapolation", value=False)
                    n_extrapolation_steps_input = gr.Number(label="Steps", value=5, minimum=1, maximum=50)
                    extrapolation_step_size_input = gr.Number(label="Step Size", info="Optional")

        # Configuration Management Connections
        def handle_save_config(all_configs):
            filepath, error = save_app_configuration(all_configs)
            if error:
                return gr.update(visible=False), gr.update(value=f"🔴 Error: {error}", visible=True)
            return gr.update(value=filepath, visible=True), gr.update(value=f"✅ Configuration saved to: {filepath}", visible=True)

        save_config_button.click(
            fn=handle_save_config,
            inputs=[curve_configs_state],
            outputs=[download_config_link, config_management_status]
        )


        def handle_load_config(file_obj, current_curve_configs, current_curve_names, current_idx):
            if file_obj is None:
                return current_curve_configs, current_curve_names, current_idx, "", gr.update(value="No file selected for loading.", visible=True), gr.update(visible=False), *[gr.update() for _ in range(17)]
            
            filepath = file_obj.name
            loaded_configs, error = load_app_configuration(filepath)
            
            if error:
                return current_curve_configs, current_curve_names, current_idx, "", gr.update(value=f"🔴 Error loading: {error}", visible=True), gr.update(visible=False), *[gr.update() for _ in range(17)]
            
            if not loaded_configs:
                return current_curve_configs, current_curve_names, current_idx, "", gr.update(value="ℹ️ Loaded configuration is empty or invalid. No changes applied.", visible=True), gr.update(visible=False), *[gr.update() for _ in range(17)]

            new_curve_names = [config.get('name', f'Dataset {i+1}') for i, config in enumerate(loaded_configs)]
            new_current_idx = 0 if new_curve_names else 0
            new_curve_selector_value = new_curve_names[new_current_idx] if new_curve_names else None

            if loaded_configs:
                first_config = loaded_configs[new_current_idx]
                ui_updates = (
                    new_current_idx,
                    gr.Textbox(value=first_config.get('x_text', '')),
                    gr.Textbox(value=first_config.get('y_text', '')),
                    gr.Textbox(value=first_config.get('z_text', '')),
                    gr.Slider(value=first_config.get('degree', 3)),
                    gr.ColorPicker(value=first_config.get('data_color', '#1f77b4')),
                    gr.ColorPicker(value=first_config.get('fit_color', '#ff7f0e')),
                    gr.Dropdown(value=first_config.get('data_marker', 'circle')),
                    gr.Dropdown(value=first_config.get('fit_line_style', 'solid')),
                    gr.Dropdown(value=first_config.get('fit_type', 'polynomial')),
                    gr.Checkbox(value=first_config.get('show_fit', True)),
                    gr.Checkbox(value=first_config.get('show_data', True)),
                    gr.Textbox(value=first_config.get('x_errors_text', '')),
                    gr.Textbox(value=first_config.get('y_errors_text', '')),
                    gr.Checkbox(value=first_config.get('show_error_bars', False)),
                    gr.Checkbox(value=first_config.get('visible', True)),
                    gr.Checkbox(value=first_config.get('force_3d', False)),
                    f"✅ Loaded {len(loaded_configs)} dataset(s). Active: {new_curve_selector_value}"
                )
            else:
                ui_updates = tuple([gr.update() for _ in range(17)]) + ("No datasets loaded",)

            return loaded_configs, new_curve_names, new_current_idx, gr.Dropdown(choices=new_curve_names, value=new_curve_selector_value), gr.update(value=f"✅ Configuration loaded from: {filepath}", visible=True), gr.update(visible=False), *ui_updates[1:] # Unpack starting from x_input

        load_config_file_input.upload(
            fn=handle_load_config,
            inputs=[load_config_file_input, curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[
                curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, config_management_status, download_config_link,
                # Outputs to update the UI for the first loaded curve (matches switch_curve outputs)
                x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox, curve_status # curve_status is updated by handle_load_config directly
            ]
        )

        # Dataset management connections
        new_curve_btn.click(
            fn=create_new_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        )

        duplicate_curve_btn.click(
            fn=duplicate_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        )

        delete_curve_btn.click(
            fn=delete_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        )

        curve_selector.select(
            fn=switch_curve,
            inputs=[curve_configs_state, curve_names_state, curve_selector],
            outputs=[
                current_curve_idx_state, x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox, # Added force_3d_checkbox to outputs
                curve_status
            ]
        )

        # File upload handling
        file_input.change(
            fn=handle_file_change,
            inputs=[file_input],
            outputs=[x_column_dropdown, y_column_dropdown, z_column_dropdown, file_dataframe_state, x_input, y_input, file_status_output]
        )

        
        
        # Consolidate all inputs for the combined plot
        # These are inputs that affect the overall plot, not individual curve configs
        # Individual curve configs are passed via curve_configs_state
        combined_plot_inputs = [
            curve_configs_state, # This now carries all individual curve settings including force_3d
            plot_options_checkbox, 
            xaxis_scale_dropdown, 
            yaxis_scale_dropdown, 
            special_points_checkbox, # This is for the combined plot, if applicable
            x_derivative_input, # This might be less relevant for combined, or apply to first curve?
            custom_x_label_input, 
            custom_y_label_input, 
            custom_title_input, 
            font_family_dropdown, 
            title_font_size_slider, 
            axes_font_size_slider, 
            legend_font_size_slider, 
            font_color_picker, 
            area_start_x_input, # Area for combined or first curve?
            area_end_x_input, 
            show_area_checkbox, 
            n_extrapolation_steps_input, # Extrapolation for combined or first curve?
            extrapolation_step_size_input, 
            show_extrapolation_checkbox
        ]
        
        update_button.click(
            fn=update_combined_plot,
            inputs=combined_plot_inputs, # Use the consolidated list
            outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output]
        )

        # Auto-save current dataset configuration
        # This function is triggered when individual curve settings are changed
        # It saves the current state of UI elements for the active curve into curve_configs_state
        
        # Define the list of UI components that, when changed, should trigger an auto-save
        # of the currently selected dataset's configuration.
        auto_save_trigger_components = [
            x_input, y_input, z_input, degree_slider, 
            data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
            fit_type_dropdown, show_fit_checkbox, show_data_checkbox, 
            x_errors_input, y_errors_input, show_error_bars_checkbox, 
            visible_checkbox, force_3d_checkbox, # force_3d is part of individual curve config
            # File-related dropdowns also need to trigger save if they change the active curve's config
            x_column_dropdown, y_column_dropdown, z_column_dropdown 
        ]

        # Define the list of all inputs needed by save_current_curve_config
        # This should include all UI elements that define a single curve's state.
        save_config_inputs = [
            curve_configs_state, current_curve_idx_state, 
            x_input, y_input, z_input, degree_slider,
            data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
            fit_type_dropdown, show_fit_checkbox, show_data_checkbox, 
            x_errors_input, y_errors_input, show_error_bars_checkbox, 
            visible_checkbox, force_3d_checkbox,
            file_dataframe_state, # This state holds the dataframe for the *currently selected* file if one was uploaded
            x_column_dropdown, y_column_dropdown, z_column_dropdown
        ]
        
        for component in auto_save_trigger_components:
            component.change(
                fn=save_current_curve_config,
                inputs=save_config_inputs, # Use the consolidated list
                outputs=[curve_configs_state] # Only curve_configs_state is modified by saving
            )

        # Initial load - uses the same inputs as the update_button
        demo.load(
            fn=update_combined_plot,
            inputs=combined_plot_inputs, # Use the consolidated list
            outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output]
        )

    return demo

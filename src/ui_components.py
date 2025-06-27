# --- START OF FILE ui_components.py ---

"""
Gradio User Interface Module

This module contains all the Gradio UI components and layout definitions.
"""

import gradio as gr
from .data_input import handle_file_change
from .dataset_management import (
    create_new_curve, duplicate_curve, delete_curve, switch_curve, save_current_curve_config,
    save_app_configuration, load_app_configuration
)
# Removed update_plot_and_fit as it is not used in this file
from .plotting import update_combined_plot, update_slider_only


def create_gradio_interface():
    """Create and configure the Gradio interface."""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=CMU+Serif&display=swap');
        
            /* Цвет фона чекбокса */
            input[type="checkbox"] {
                accent-color: #ff7f0e !important; 
                background-color: #ff7f0e !important;
            }
            body, .gradio-container {
                background: rgba(0, 97, 255,0.15) !important; /* CornflowerBlue with alpha=0.5 */ }
            .custom-btn button {
                background: #ef7500 !important;
                color: green !important;
                border: none !important;
            }
            </style>
            """
        )
        gr.Markdown(
            "# 📊 Multi-Dataset Polynomial Fitting & Analysis Tool\n"
            "Upload files or input data manually to fit polynomials, analyze statistics, and visualize results. "
            "Each dataset can have its own polynomial fit, styling, and configuration. "
            "💡 **Z-axis support**: Add optional Z values for color mapping in both manual input and file upload."
            "📈 **Log-scale support**: You can use logarithmic axes for X and Y if all data points are non-zero."

        )

        # Dataset management state
        curve_configs_state = gr.State([{
            'name': 'Dataset 1',
            'x_text': '0, 1, 2, 3, 4, 5, 6',
            'y_text': '-1.2, 0, 0.6, 1.2, 2.4, 5.0, 9.8',
            'z_text': '',
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
            'file_df': None,
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
                        force_3d_checkbox = gr.Checkbox(label="Attempt 3D Surface Plot (if Z data present)",
                                                             value=False, interactive=True, elem_classes="my-checkbox")

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
                # Add the dataset name input here
                with gr.Column():
                    dataset_name_input = gr.Textbox(label="Dataset Name", interactive=True)
                    with gr.Row():
                        update_name_btn = gr.Button("✔", size="sm",  elem_classes="custom-btn")  # маленькая кнопка
                        placeholder_one = gr.Button("✔",variant="primary", size="sm", visible=False)  # Placeholder for alignment

                with gr.Row():
                    show_data_checkbox = gr.Checkbox(label="Show Data Points", value=True, elem_classes="my-checkbox")
                    show_fit_checkbox = gr.Checkbox(label="Show Polynomial Fit", value=True, elem_classes="my-checkbox")
                    connect_points_checkbox = gr.Checkbox(label="Connect Data Points", value=False, elem_classes="my-checkbox")

                visible_checkbox = gr.Checkbox(label="Visible on Plot", value=True, visible= False)

                degree_slider = gr.Slider(
                    minimum=0, maximum=10, step=1, value=3,
                    label="Polynomial Degree"
                )

                fit_type_dropdown = gr.Dropdown(
                    choices=["polynomial"],
                    value="polynomial",
                    label="Fitting Type"
                )

                

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

                

            with gr.Column(scale=2):
                gr.Markdown("### 📈 Multi-Dataset Visualization")
                plot_output = gr.Plot(label="Combined Plot", show_label=True)

                # Main update button for the combined plot
                update_button = gr.Button("🚀 Update Multi-Dataset Plot", size="lg", elem_classes="custom-btn")

                equation_output = gr.Markdown(
                    label="Plot Status",
                    #interactive=False,
                    #lines=2,
                    render=True
                )

                with gr.Tabs():
                    with gr.TabItem("📊 Statistics"):
                        statistics_output = gr.Markdown(
                            label="Statistical Analysis",
                            #interactive=False,
                            #lines=15,
                            show_label=False,
                            render=True     
                        )

                    with gr.TabItem("📐 Area Analysis"):
                        area_output = gr.Markdown(
                            label="Area Under Curve",
                            #interactive=False,
                            #lines=10,
                            show_label=False,
                            render=True
                        )

                    with gr.TabItem("🔮 Extrapolation"):
                        extrapolation_output = gr.Markdown(
                            label="Prediction Results",
                            #interactive=False,
                            #lines=12,
                            show_label=False,
                            render=True
                        )
                # Configuration Management Section
                gr.Markdown("### 💾 Configuration Management")
                with gr.Column():
                    save_config_button = gr.Button("Save Configuration", size="sm")
                    load_config_file_input = gr.File(
                        label="Load Configuration File",
                        file_types=[".json"],
                        type="filepath",
                        scale=2
                    )
                config_management_status = gr.Textbox(label="Config Status", interactive=False, lines=2, visible=False)
                download_config_link = gr.File(label="Download Saved Configuration", interactive=False, visible=False)


                # Error bars
                show_error_bars_checkbox = gr.Checkbox(label="Show Error Bars (Enter value for every data point)", value=False, elem_classes="my-checkbox")
                x_errors_input = gr.Textbox(label="X Uncertainties", placeholder="Optional", lines=1)
                y_errors_input = gr.Textbox(label="Y Uncertainties", placeholder="Optional", lines=1)

        with gr.Row("Plot Customization and Analysis Tools"):
            with gr.Column():
                with gr.Column():
                    gr.Markdown("**Display Options**")
                    plot_options_checkbox = gr.CheckboxGroup(
                        ["Show Grid", "Show Axes Labels", "Show Title"],
                        label="Plot Options",
                        value=["Show Grid", "Show Axes Labels", "Show Title"], elem_classes="my-checkbox"
                    )
                    xaxis_scale_dropdown = gr.Dropdown(choices=["linear", "log"], value="linear", label="X-axis Scale")
                    yaxis_scale_dropdown = gr.Dropdown(choices=["linear", "log"], value="linear", label="Y-axis Scale")

                with gr.Column():
                    gr.Markdown("**Labels & Title**")
                    custom_x_label_input = gr.Textbox(label="X-axis Label", placeholder="X values")
                    custom_y_label_input = gr.Textbox(label="Y-axis Label", placeholder="Y values")
                    custom_title_input = gr.Textbox(label="Plot Title", placeholder="Data and Polynomial Fit")

                with gr.Column():
                    gr.Markdown("**Font Settings**")
                    font_family_dropdown = gr.Dropdown(
                        choices=["Arial", "Times New Roman", "Courier New", "Georgia", "CMU Serif"],
                        value="Arial", label="Font Family"
                    )
                    title_font_size_slider = gr.Slider(8, 54, 16, label="Title Font Size")
                    axes_font_size_slider = gr.Slider(8, 32, 12, label="Axes Font Size")
                    legend_font_size_slider = gr.Slider(8, 32, 10, label="Legend Font Size")
                    font_color_picker = gr.ColorPicker(label="Font Color", value="#000000")

            with gr.Column():
                with gr.Column():
                    gr.Markdown("**Special Points**")
                    special_points_checkbox = gr.CheckboxGroup(
                        ["Show Extrema", "Show X-Intercepts", "Show Y-Intercept"],
                        label="Special Points", value=[], elem_classes="my-checkbox"
                    )
                    x_derivative_input = gr.Number(label="X for f'(x)", info="Optional")
                    derivative_output_text = gr.Textbox(label="Derivative", interactive=False, lines=1)

                with gr.Column():
                    gr.Markdown("**Area Under Curve**")
                    show_area_checkbox = gr.Checkbox(label="Show Area", value=False, elem_classes="my-checkbox")
                    area_start_x_input = gr.Number(label="Start X")
                    area_end_x_input = gr.Number(label="End X")

                with gr.Column():
                    gr.Markdown("**Extrapolation**")
                    show_extrapolation_checkbox = gr.Checkbox(label="Show Extrapolation", value=False, elem_classes="my-checkbox")
                    n_extrapolation_steps_input = gr.Number(label="Steps", value=5, minimum=1, maximum=50)
                    extrapolation_step_size_input = gr.Number(label="Step Size", info="Optional")

        # Configuration Management Connections
        def handle_save_config(all_configs):
            filepath, error = save_app_configuration(all_configs)
            if error:
                # Не показываем download, только статус
                return None, gr.update(value=f"🔴 Error: {error}", visible=True)
            # Показываем download и статус
            return gr.update(value=filepath, visible=True), gr.update(value=f"✅ Configuration saved to: {filepath}", visible=True)

        save_config_button.click(
            fn=handle_save_config,
            inputs=[curve_configs_state],
            outputs=[download_config_link, config_management_status]
        )

        def handle_load_config(file_obj, current_curve_configs, current_curve_names, current_idx):
            if file_obj is None:
                return (
                    current_curve_configs, current_curve_names, current_idx, gr.update(),
                    gr.update(value="No file selected for loading.", visible=True),  # config status
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update()
                )

            filepath = file_obj.name
            loaded_configs, error = load_app_configuration(filepath)

            if error:
                return (
                    current_curve_configs, current_curve_names, current_idx, gr.update(),
                    gr.update(value=f"🔴 Error loading: {error}", visible=True),  # config status
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update()
                )

            if not loaded_configs:
                return (
                    current_curve_configs, current_curve_names, current_idx, gr.update(),
                    gr.update(value="ℹ️ Loaded configuration is empty or invalid. No changes applied.", visible=True),  # config status
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update()
                )

            new_curve_names = [config.get('name', f'Dataset {i+1}') for i, config in enumerate(loaded_configs)]
            new_current_idx = 0 if new_curve_names else 0
            new_curve_selector_value = new_curve_names[new_current_idx] if new_curve_names else None

            if loaded_configs:
                first_config = loaded_configs[new_current_idx]
                return (
                    loaded_configs,
                    new_curve_names,
                    new_current_idx,
                    gr.Dropdown(choices=new_curve_names, value=new_curve_selector_value),
                    gr.update(value=f"✅ Configuration loaded from: {filepath}", visible=True),  # config status
                    gr.Textbox(value=first_config.get('name', '')),
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
                return (
                    loaded_configs, new_curve_names, new_current_idx, gr.Dropdown(choices=new_curve_names, value=new_curve_selector_value),
                    gr.update(value=f"✅ Configuration loaded from: {filepath}", visible=True),  # config status
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    "No datasets loaded"
                )


        load_config_file_input.upload(
            fn=handle_load_config,
            inputs=[load_config_file_input, curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[
                curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, config_management_status,
                # Outputs to update the UI for the first loaded curve (matches switch_curve outputs + dataset_name_input)
                dataset_name_input, x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox, curve_status
            ]
        )

        # Dataset management connections
        # These functions create/delete/duplicate and need to update the dropdown list itself
        # Use .then() to trigger switch_curve after the dataset list and current index are updated
        new_curve_btn.click(
            fn=create_new_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        ).then(
             # After creating/duplicating/deleting, switch to the new/selected curve
             # Use the value set for curve_selector by the previous function's outputs
             fn=switch_curve,
             inputs=[curve_configs_state, curve_names_state, curve_selector],
             outputs=[
                current_curve_idx_state, # Update state directly
                dataset_name_input, x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox, # Added force_3d_checkbox to outputs
                curve_status
             ]
        )

        duplicate_curve_btn.click(
            fn=duplicate_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        ).then(
             # After duplicating, switch to the new duplicated curve
             fn=switch_curve,
             inputs=[curve_configs_state, curve_names_state, curve_selector],
             outputs=[
                current_curve_idx_state,
                dataset_name_input, x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox,
                curve_status
             ]
        )

        delete_curve_btn.click(
            fn=delete_curve,
            inputs=[curve_configs_state, curve_names_state, current_curve_idx_state],
            outputs=[curve_configs_state, curve_names_state, current_curve_idx_state, curve_selector, curve_status]
        ).then(
             # After deleting, switch to the new active curve
             fn=switch_curve,
             inputs=[curve_configs_state, curve_names_state, curve_selector],
             outputs=[
                current_curve_idx_state,
                dataset_name_input, x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox,
                curve_status
             ]
        )


        # When switching datasets, update all the config UI elements
        # switch_curve returns: curve_idx, dataset_name, x_input, y_input, z_input, degree_slider, data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown, fit_type_dropdown, show_fit_checkbox, show_data_checkbox, x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox, force_3d_checkbox, curve_status
        curve_selector.change(
            fn=switch_curve,
            inputs=[curve_configs_state, curve_names_state, curve_selector],
            outputs=[
                current_curve_idx_state, # First output is the updated state
                dataset_name_input, # Added dataset_name_input here
                x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
                force_3d_checkbox, # Added force_3d_checkbox to outputs
                curve_status
            ]
        )

        # File upload handling
        # Note: handle_file_change updates x/y/z_input and file_dataframe_state
        # When file changes, it should also trigger a save of the *current* curve state
        # and then potentially update the dropdowns for the *current* curve's config.
        # This is handled by the save_current_curve_config triggers below.
        file_input.change(
            fn=handle_file_change,
            inputs=[file_input],
            outputs=[x_column_dropdown, y_column_dropdown, z_column_dropdown, file_dataframe_state, x_input, y_input, file_status_output]
        )

        # When column selections change, these need to trigger a save of the current curve's config
        # to store the selected column names and the associated file_df state.
        # Use the updated save_config_inputs which includes dataset_name_input
        save_config_inputs_with_name = [ # New list including dataset_name_input
                curve_configs_state, current_curve_idx_state,
                dataset_name_input, # Added
                x_input, y_input, z_input, degree_slider,
                data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
                x_errors_input, y_errors_input, show_error_bars_checkbox,
                visible_checkbox, force_3d_checkbox,
                file_dataframe_state,
                x_column_dropdown, y_column_dropdown, z_column_dropdown
            ]

        # Connect the change events for the column dropdowns to save the state
        x_column_dropdown.change(
            fn=save_current_curve_config,
            inputs=save_config_inputs_with_name,
            outputs=[curve_configs_state]
        )
        y_column_dropdown.change(
            fn=save_current_curve_config,
            inputs=save_config_inputs_with_name,
            outputs=[curve_configs_state]
        )
        z_column_dropdown.change(
            fn=save_current_curve_config,
            inputs=save_config_inputs_with_name,
            outputs=[curve_configs_state]
        )


        

        # Consolidate all inputs for the combined plot
        combined_plot_inputs = [
            curve_configs_state,
            plot_options_checkbox,
            xaxis_scale_dropdown,
            yaxis_scale_dropdown,
            special_points_checkbox,
            x_derivative_input,
            custom_x_label_input,
            custom_y_label_input,
            custom_title_input,
            font_family_dropdown,
            title_font_size_slider,
            axes_font_size_slider,
            legend_font_size_slider,
            font_color_picker,
            area_start_x_input,
            area_end_x_input,
            show_area_checkbox,
            n_extrapolation_steps_input,
            extrapolation_step_size_input,
            show_extrapolation_checkbox,
            connect_points_checkbox
        ]

        update_button.click(
            fn=update_combined_plot,
            inputs=combined_plot_inputs,
            outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output]
        )

        # Auto-save current dataset configuration
        # This function is triggered when individual curve settings are changed
        # It saves the current state of UI elements for the active curve into curve_configs_state

        # Define the list of UI components that, when changed, should trigger an auto-save
        # of the currently selected dataset's configuration.
        # Added dataset_name_input here
        auto_save_trigger_components = [
            dataset_name_input, # Added
            x_input, y_input, z_input, degree_slider,
            data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
            fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
            x_errors_input, y_errors_input, show_error_bars_checkbox,
            visible_checkbox, force_3d_checkbox,
            # File-related dropdowns are handled by their specific change events above
            # file_dataframe_state doesn't have a change event
        ]

        def handle_dataset_name_change(curve_names, curve_configs, current_idx, new_name):
            updated_names = list(curve_names)
            updated_names[current_idx] = new_name
            updated_configs = list(curve_configs)
            updated_configs[current_idx] = dict(updated_configs[current_idx])
            updated_configs[current_idx]['name'] = new_name

            # value всегда только по индексу!
            if 0 <= current_idx < len(updated_names):
                value = updated_names[current_idx]
            elif updated_names:
                value = updated_names[0]
            else:
                value = None

            return (
                updated_names,
                updated_configs,
                gr.update(choices=updated_names, value=value),
            )

        update_name_btn.click(
            fn=handle_dataset_name_change,
            inputs=[curve_names_state, curve_configs_state, current_curve_idx_state, dataset_name_input],
            outputs=[curve_names_state, curve_configs_state, curve_selector]
        )

        # Use the updated save_config_inputs list
        # The file column dropdowns are *inputs* to save_current_curve_config,
        # but their *changes* are handled by the specific triggers above to ensure
        # the change().then() sequence works correctly.
        save_config_inputs_for_triggers = [
             curve_configs_state, current_curve_idx_state,
             dataset_name_input, x_input, y_input, z_input, degree_slider,
             data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
             fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
             x_errors_input, y_errors_input, show_error_bars_checkbox,
             visible_checkbox, force_3d_checkbox,
             file_dataframe_state, x_column_dropdown, y_column_dropdown, z_column_dropdown
        ]

        # Connect the auto-save triggers (excluding the file column dropdowns as they have their own trigger chain)
        for component in [c for c in auto_save_trigger_components if c not in [x_column_dropdown, y_column_dropdown, z_column_dropdown]]:
            component.change(
                fn=save_current_curve_config,
                inputs=save_config_inputs_for_triggers,
                outputs=[curve_configs_state]
            )


        # Initial load - uses the same inputs as the update_button
        # Also needs to populate the initial dataset configuration UI elements
        initial_ui_load_outputs = [
            plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output,
            # Initial state of current curve config UI elements (based on the first dataset)
            dataset_name_input, x_input, y_input, z_input, degree_slider,
            data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
            fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
            x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
            force_3d_checkbox, curve_status
        ]

        # Create a function to handle the initial load of UI elements
        def initial_load_ui_update(curve_configs, curve_names):
            if not curve_configs:
                # This should ideally never happen with the default state
                # Return a tuple matching the outputs with default/empty values or gr.update()
                # Length of initial_ui_load_outputs is 6 (plot) + 18 (config) = 24
                return (
                    None, "", "", "", "", "", # plot, eq, deriv, stats, area, extrap
                    gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=3), # name, x, y, z, degree
                    gr.update(value="#1f77b4"), gr.update(value="#ff7f0e"), gr.update(value="circle"), gr.update(value="solid"), # colors, marker, line
                    gr.update(value="polynomial"), gr.update(value=True), gr.update(value=True), # fit_type, show_fit, show_data
                    gr.update(value=""), gr.update(value=""), gr.update(value=False), gr.update(value=True), # errors, show_err, visible
                    gr.update(value=False), # force_3d
                    "No datasets available" # curve_status
                )

            # switch_curve returns: curve_idx, dataset_name, x_input, y_input, z_input, degree_slider, data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown, fit_type_dropdown, show_fit_checkbox, show_data_checkbox, x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox, force_3d_checkbox, curve_status
            initial_switch_outputs = switch_curve(curve_configs, curve_names, curve_names[0])

            # Combine plot outputs (empty initially) with the initial switch outputs
            initial_plot_outputs = (None, "", "", "", "", "") # plot, equation, derivative, stats, area, extrapolation

            # The outputs for initial_ui_load_outputs are the 6 plot outputs + the 18 switch_curve UI outputs (excluding the first element which is the state update)
            combined_outputs = initial_plot_outputs + initial_switch_outputs[1:] # Exclude current_curve_idx_state from switch_curve outputs

            return combined_outputs

        # Initial UI load (populate config fields for the first dataset)
        demo.load(
            fn=initial_load_ui_update,
            inputs=[curve_configs_state, curve_names_state],
            outputs=initial_ui_load_outputs,
            queue=False
        ).then(
             # After populating the UI, trigger the first plot update
             fn=update_combined_plot,
             inputs=combined_plot_inputs,
             outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output],
             queue=False
        )


    return demo
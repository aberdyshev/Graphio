"""
Plotting and Visualization Module

This module handles all plotting operations, visualization, and plot generation.
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import warnings
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from scipy.interpolate import griddata

from .data_input import parse_input_data
from .math_functions import fit_alternative_model, calculate_extrapolation, calculate_area_under_curve, format_polynomial
from .statistics import calculate_statistics, format_statistics_output, format_area_output, format_extrapolation_output


def update_slider_only(x_text: str, y_text: str):
    """Updates only the degree slider based on data points."""
    import gradio as gr
    
    try:
        x_data, x_error = parse_input_data(x_text)
        y_data, y_error = parse_input_data(y_text)
        
        if x_error or y_error or len(x_data) == 0 or len(y_data) == 0 or len(x_data) != len(y_data):
            return gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")
        
        max_degree = max(0, len(x_data) - 1)
        return gr.Slider(minimum=0, maximum=max_degree, step=1, value=min(3, max_degree), label="Polynomial Degree (n)")
    except Exception:
        return gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")


def _is_suitable_for_3d_surface(x_data, y_data, z_data):
    """
    Check if the data is suitable for 3D surface plotting.
    
    Args:
        x_data: Array of X coordinates
        y_data: Array of Y coordinates 
        z_data: Array of Z values
        
    Returns:
        bool: True if data is suitable for 3D surface plotting
    """
    if z_data is None or len(z_data) == 0:
        return False
    
    # Check if we have enough data points for a meaningful surface
    if len(x_data) < 9:  # Need at least 3x3 grid
        return False
    
    # Check if we have some variation in X and Y coordinates
    x_unique = len(np.unique(x_data))
    y_unique = len(np.unique(y_data))
    
    # Need at least 3 unique values in both X and Y to create a surface
    return x_unique >= 3 and y_unique >= 3


def _create_3d_surface_plot(x_data, y_data, z_data, x_errors=None, y_errors=None, 
                           data_color="#1f77b4", custom_x_label="X", 
                           custom_y_label="Y", custom_title="3D Surface Plot"):
    """
    Create a 3D surface plot from the provided data.
    
    Args:
        x_data: Array of X coordinates
        y_data: Array of Y coordinates
        z_data: Array of Z values
        x_errors: Optional X error values
        y_errors: Optional Y error values
        data_color: Color for the surface
        custom_x_label: Label for X axis
        custom_y_label: Label for Y axis
        custom_title: Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: 3D surface plot
    """
    fig = go.Figure()
    
    # Create grid for surface interpolation
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    # Create regular grid
    grid_resolution = max(20, int(np.sqrt(len(x_data))))
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate Z values onto the regular grid
    points = np.column_stack((x_data, y_data))
    Z_grid = griddata(points, z_data, (X_grid, Y_grid), method='cubic', fill_value=np.nan)
    
    # Handle NaN values by using linear interpolation as fallback
    if np.isnan(Z_grid).any():
        Z_grid_linear = griddata(points, z_data, (X_grid, Y_grid), method='linear', fill_value=np.nan)
        Z_grid = np.where(np.isnan(Z_grid), Z_grid_linear, Z_grid)
    
    # Create surface plot
    fig.add_trace(go.Surface(
        x=X_grid,
        y=Y_grid,
        z=Z_grid,
        colorscale='Viridis',
        name='Surface',
        showscale=True,
        colorbar=dict(title="Z Values")
    ))
    
    # Add original data points as scatter
    fig.add_trace(go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle'
        ),
        name='Original Data Points'
    ))
    
    # Update layout for 3D
    fig.update_layout(
        title=custom_title,
        scene=dict(
            xaxis_title=custom_x_label,
            yaxis_title=custom_y_label,
            zaxis_title="Z Values",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    
    return fig

def update_plot_and_fit(
    x_text, y_text, degree, plot_options,
    xaxis_scale, yaxis_scale, special_points_show, x_for_derivative,
    data_color, fit_color, data_marker, fit_line_style,
    custom_x_label, custom_y_label, custom_title,
    font_family, title_font_size, axes_font_size, legend_font_size, font_color,
    area_start_x, area_end_x, show_area,
    fit_type, n_extrapolation_steps, extrapolation_step_size, show_extrapolation,
    x_errors_text, y_errors_text, show_error_bars,
    file_df, x_col_name, y_col_name, z_col_name, # Existing file/column params
    force_3d=False, connect_points=False  # New parameter for explicit 3D plot request
):
    """Enhanced function with file upload, column selection, Z-component, and explicit 3D plot support."""
    import gradio as gr
    
    try:
        derivative_text_output = ""
        statistics_output = ""
        area_output = ""
        extrapolation_output = ""
        info_msg = ""
        equation_str = ""
        fit_params_str = ""
        area_str = ""
        
        # Create default slider for error cases
        default_slider = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")

        x_data = np.array([])
        y_data = np.array([])
        z_data = None
        data_source_message = ""

        # --- Data Source Determination ---
        if file_df is not None and x_col_name and y_col_name:
            try:
                # File-based data processing
                x_data_series = pd.to_numeric(file_df[x_col_name], errors='coerce')
                y_data_series = pd.to_numeric(file_df[y_col_name], errors='coerce')
                
                # Handle Z column if specified
                z_data_series = None
                if z_col_name and z_col_name in file_df.columns:
                    z_data_series = pd.to_numeric(file_df[z_col_name], errors='coerce')
                
                # Create combined dataframe without NaN values
                if z_data_series is not None:
                    df = pd.DataFrame({'x': x_data_series, 'y': y_data_series, 'z': z_data_series})
                    original_rows = len(df)
                    df = df.dropna()
                    valid_rows = len(df)
                    x_data = df['x'].to_numpy()
                    y_data = df['y'].to_numpy()
                    z_data = df['z'].to_numpy()
                else:
                    df = pd.DataFrame({'x': x_data_series, 'y': y_data_series})
                    original_rows = len(df)
                    df = df.dropna()
                    valid_rows = len(df)
                    x_data = df['x'].to_numpy()
                    y_data = df['y'].to_numpy()
                
                # Create informative message
                cols_used = f"X='{x_col_name}', Y='{y_col_name}'"
                if z_col_name:
                    cols_used += f", Z='{z_col_name}'"
                
                data_source_message = f"📊 **File Data**: Loaded {valid_rows} valid data points from file.\n📋 **Columns**: {cols_used}"
                
                if valid_rows < original_rows:
                    dropped = original_rows - valid_rows
                    data_source_message += f"\n⚠️ **Dropped**: {dropped} rows due to non-numeric/missing values."

                # File data ignores manual error inputs
                x_errors = None
                y_errors = None
                error_info = "ℹ️ Error bars from manual input are ignored when using file data." if show_error_bars else ""

            except Exception as e:
                return None, f"�� **File Processing Error**: {str(e)}", "", "", statistics_output, "", default_slider
                
        else:
            # Parse from manual text inputs
            x_data_parsed, x_error_msg = parse_input_data(x_text)
            y_data_parsed, y_error_msg = parse_input_data(y_text)

            if x_error_msg:
                return None, f"🔴 **X Data Error**: {x_error_msg}", "", "", "", "", default_slider
            if y_error_msg:
                return None, f"🔴 **Y Data Error**: {y_error_msg}", "", "", "", "", default_slider
            
            x_data, y_data = x_data_parsed, y_data_parsed
            data_source_message = "📝 **Manual Input**: Using data from text fields."

            # Parse Z data for manual input
            if z_col_name:  # This would be from z_input in manual mode
                z_parsed, z_error = parse_input_data(z_col_name)  # z_col_name is actually z_text in this context
                if not z_error and len(z_parsed) > 0:
                    if len(z_parsed) == len(x_data):
                        z_data = z_parsed
                    else:
                        data_source_message += f"\n⚠️ Z data length ({len(z_parsed)}) doesn't match X/Y data length ({len(x_data)}), ignoring Z data."

            # Parse error data for manual input
            x_errors = None
            y_errors = None
            error_info = ""
            
            if show_error_bars:
                if x_errors_text and x_errors_text.strip():
                    x_errors_data, x_errors_err_msg = parse_input_data(x_errors_text)
                    if x_errors_err_msg:
                        error_info += f"⚠️ X errors: {x_errors_err_msg}\n"
                    elif len(x_errors_data) != len(x_data):
                        error_info += f"⚠️ X errors length ({len(x_errors_data)}) ≠ data length ({len(x_data)})\n"
                    else:
                        x_errors = x_errors_data
                
                if y_errors_text and y_errors_text.strip():
                    y_errors_data, y_errors_err_msg = parse_input_data(y_errors_text)
                    if y_errors_err_msg:
                        error_info += f"⚠️ Y errors: {y_errors_err_msg}\n"
                    elif len(y_errors_data) != len(y_data):
                        error_info += f"⚠️ Y errors length ({len(y_errors_data)}) ≠ data length ({len(y_data)})\n"
                    else:
                        y_errors = y_errors_data

        # --- Input Validation ---
        if len(x_data) == 0:
            return None, "🔴 **Empty Data**: No valid X values found.", "", "", "", "", default_slider
        if len(y_data) == 0:
            return None, "🔴 **Empty Data**: No valid Y values found.", "", "", "", "", default_slider
        
        if len(x_data) != len(y_data):
            return None, f"🔴 **Data Length Mismatch**: X has {len(x_data)} values, Y has {len(y_data)} values.\n*Both arrays must have the same length.*", "", "", "", "", default_slider

        # Calculate statistics
        stats, _ = calculate_statistics(x_data, y_data)
        if stats:
            statistics_output = format_statistics_output(stats)
            statistics_output = f"{data_source_message}\n\n{statistics_output}"
        else:
            statistics_output = data_source_message

        # Add error bar information
        if error_info:
            statistics_output += f"\n\n📊 **Error Bar Info:**\n{error_info.strip()}"
        elif show_error_bars and (x_errors is not None or y_errors is not None):
            error_summary = []
            if x_errors is not None:
                error_summary.append(f"X errors: Mean = {np.mean(x_errors):.4f}")
            if y_errors is not None:
                error_summary.append(f"Y errors: Mean = {np.mean(y_errors):.4f}")
            if error_summary:
                statistics_output += f"\n\n📊 **Error Bar Info:**\n• " + "\n• ".join(error_summary)

        # Calculate max degree and validate for polynomial
        max_degree = max(0, len(x_data) - 1)
        if degree > max_degree:
            degree = max_degree
        
        updated_slider = gr.Slider(minimum=0, maximum=max_degree, step=1, value=degree, label="Polynomial Degree (n)")
        
        # --- Model Fitting based on selected type ---
        try:
            if fit_type == "polynomial":
                # Traditional polynomial fitting
                if len(x_data) > 1 and degree > 0:
                    coeffs = np.polyfit(x_data, y_data, degree)
                    fit_func = np.poly1d(coeffs)
                    equation_str = format_polynomial(coeffs, degree)
                    model_params = None
                    
                    # Calculate R-squared
                    y_pred = fit_func(x_data)
                    ss_res = np.sum((y_data - y_pred) ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    if r_squared >= 0.9:
                        info_msg = f"✅ **Polynomial Fit** (degree {degree})\n🎯 **Excellent Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.7:
                        info_msg = f"✅ **Polynomial Fit** (degree {degree})\n👍 **Good Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.5:
                        info_msg = f"✅ **Polynomial Fit** (degree {degree})\n👌 **Moderate Fit**: R² = {r_squared:.4f}"
                    else:
                        info_msg = f"✅ **Polynomial Fit** (degree {degree})\n⚠️ **Poor Fit**: R² = {r_squared:.4f}"
                else:
                    info_msg = f"ℹ️ **Single Point**: Using constant function"
                    # For single point or degree 0, use constant function
                    fit_func = np.poly1d([y_data[0]])
                    equation_str = f"f(x) = {y_data[0]:.3f}"
                    model_params = None
            else:
                # Alternative model fitting
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    try:
                        fit_func, model_params, fit_params_str = fit_alternative_model(x_data, y_data, fit_type)
                    except (RuntimeWarning, RuntimeError, ValueError) as e:
                        return None, f"🔴 **Fitting Error ({fit_type})**: {str(e)}", "", "", statistics_output, "", updated_slider
                
                if fit_func is None:
                    return None, f"🔴 **Fitting Error**: Could not fit {fit_type} model to data", "", "", statistics_output, "", updated_slider
                
                # Calculate R-squared for alternative models
                try:
                    if model_params is not None:
                        y_pred = fit_func(x_data, *model_params)
                    else:
                        y_pred = fit_func(x_data)
                    
                    ss_res = np.sum((y_data - y_pred) ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    if r_squared >= 0.9:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n�� **Excellent Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.7:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n👍 **Good Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.5:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n👌 **Moderate Fit**: R² = {r_squared:.4f}"
                    else:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n⚠️ **Poor Fit**: R² = {r_squared:.4f}"
                except:
                    info_msg = f"✅ **{fit_type.title()} Fit** completed"
                    
                # Make sure we have equation_str set for all models
                if not equation_str and fit_type != "custom":
                    equation_str = f"f(x) = {fit_type} model"

            # Generate points for the fitted curve
            if len(x_data) > 1:
                x_range = x_data.max() - x_data.min()
                if x_range == 0:
                    x_fit = np.array([x_data[0] - 1, x_data[0], x_data[0] + 1])
                else:
                    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                
                if model_params is not None:
                    y_fit = fit_func(x_fit, *model_params)
                else:
                    y_fit = fit_func(x_fit)
                    
                # Check for numerical issues in the fit
                if np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit)):
                    info_msg += "\n⚠️ **Warning**: Fitted curve contains invalid values"
                    # Filter out invalid points
                    valid_mask = np.isfinite(y_fit)
                    x_fit = x_fit[valid_mask]
                    y_fit = y_fit[valid_mask]
                    
                    if len(x_fit) == 0:
                        return None, "🔴 **Fitting Error**: All fitted values are invalid (NaN/Inf).", "", "", statistics_output, "", updated_slider
                        
            else:
                # Single point case
                x_fit = np.array([x_data[0]])
                y_fit = np.array([y_data[0]])
                fit_func = lambda x: np.full_like(x, y_data[0])
                equation_str = f"f(x) = {y_data[0]:.3f}"
                model_params = None

        except Exception as e:
            return None, f"🔴 **Fitting Error**: {str(e)}", "", "", statistics_output, "", updated_slider

        # --- Check for 3D Surface Plotting ---
        # If Z data is available and suitable for 3D surface, create 3D plot
        # Or if user explicitly requests 3D plot and Z data is present
        if z_data is not None and (force_3d or _is_suitable_for_3d_surface(x_data, y_data, z_data)):
            if not _is_suitable_for_3d_surface(x_data, y_data, z_data) and force_3d:
                info_msg += "\n⚠️ **3D Plot Warning**: Data may not be ideal for 3D surface, but attempting due to explicit request."
            try:
                fig = _create_3d_surface_plot(
                    x_data, y_data, z_data, x_errors, y_errors,
                    data_color, custom_x_label, custom_y_label, custom_title
                )
                
                # For 3D plots, we don't show polynomial fitting on the surface
                # but we can show fit info separately
                info_msg += "\n🎯 **3D Surface**: Created 3D surface plot from X, Y, Z data."
                
                # Apply layout customizations for 3D
                layout_options = {}
                
                # Font customization for 3D
                safe_font_family = font_family if font_family else "Arial"
                safe_title_size = max(8, min(32, title_font_size)) if title_font_size else 16
                safe_axes_size = max(8, min(24, axes_font_size)) if axes_font_size else 12
                safe_legend_size = max(8, min(20, legend_font_size)) if legend_font_size else 10
                
                layout_options['font'] = dict(
                    family=safe_font_family,
                    size=safe_axes_size,
                    color=font_color if font_color else "#000000"
                )
                
                if "Show Title" in plot_options:
                    layout_options['title_font'] = dict(
                        family=safe_font_family,
                        size=safe_title_size,
                        color=font_color if font_color else "#000000"
                    )
                
                layout_options['legend_font'] = dict(
                    family=safe_font_family,
                    size=safe_legend_size,
                    color=font_color if font_color else "#000000"
                )
                
                fig.update_layout(**layout_options)
                
                # For 3D plots, we still need to return all the expected outputs
                return fig, info_msg, equation_str, fit_params_str, statistics_output, area_str, updated_slider
                
            except Exception as e:
                # Fall back to 2D plotting if 3D fails
                info_msg += f"\n⚠️ **3D Error**: {str(e)}. Falling back to 2D plot."
        
        # --- Create 2D Plotly Figure ---
        fig = go.Figure()
        
        # Validate colors
        try:
            # Test if colors are valid by trying to add them to a trace
            fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color=data_color), visible=False))
            fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color=fit_color), visible=False))
            fig.data = []  # Clear test traces
        except Exception:
            data_color = "#1f77b4"  # Fallback to default blue
            fit_color = "#ff7f0e"   # Fallback to default orange

        # Plot original data with enhanced styling for Z-data
        scatter_params = {
            'x': x_data,
            'y': y_data,
            'mode': 'markers',
            'name': 'Original Data',
            'marker': dict(color=data_color, symbol=data_marker, size=8)
        }
        
        if connect_points:
        # Lomanaya 
          fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name='Connected Points',
            line=dict(color=data_color, width=1, dash='solid'),
            opacity=0.7,
            hoverinfo='skip'
        ))
        # Enhanced visualization if Z-data is available
        if z_data is not None:
            # Use Z-data for color mapping
            scatter_params['marker']['color'] = z_data
            scatter_params['marker']['colorscale'] = 'Viridis'
            scatter_params['marker']['showscale'] = True
            scatter_params['marker']['colorbar'] = dict(title="Z Values")
            scatter_params['name'] = f'Data (colored by Z)'
        
            def validate_data_length(x_data, y_data):
            

        #  Проверяет соответствие количества значений X и Y.
        #    Возвращает сообщение об ошибке, если размеры не совпадают.
            
                error_msgt = None
            
                # Определяем длины массивов
                x_len = len(x_data)
                y_len = len(y_data)
            
                # Проверяем соответствие размеров
                if x_len != y_len:
                    error_msgt = "❌ Ошибка: количество значений не совпадает!\n"
                
                    # Определяем, каких данных не хватает
                    if x_len > y_len:
                        missing_count = x_len - y_len
                        error_msgt += f" - Не хватает {missing_count} значений Y (имеется {y_len}, требуется {x_len})"
                    else:
                        missing_count = y_len - x_len
                        error_msgt += f" - Не хватает {missing_count} значений X (имеется {x_len}, требуется {y_len})"
                    
                return error_msgt


        # Add error bars if available
        if show_error_bars and (x_errors is not None or y_errors is not None):
            if x_errors is not None:
                scatter_params['error_x'] = dict(
                    type='data',
                    array=x_errors,
                    color=data_color,
                    thickness=1.5,
                    width=3
                )
            if y_errors is not None:
                scatter_params['error_y'] = dict(
                    type='data',
                    array=y_errors,
                    color=data_color,
                    thickness=1.5,
                    width=3
                )
            
            # Update name to indicate error bars
            error_types = []
            if x_errors is not None:
                error_types.append("X")
            if y_errors is not None:
                error_types.append("Y")
            base_name = scatter_params['name']
            scatter_params['name'] = f'{base_name} (±{"/".join(error_types)} errors)'

        fig.add_trace(go.Scatter(**scatter_params))

        # Plot fitted curve
        if len(x_fit) > 0 and len(y_fit) > 0:
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'{fit_type.title()} Fit',
                line=dict(color=fit_color, dash=fit_line_style, width=2)
            ))

        # Add extrapolation if requested
        if show_extrapolation:
            try:
                step_size = extrapolation_step_size if extrapolation_step_size and extrapolation_step_size > 0 else None
                x_extrap, y_extrap, extrap_error = calculate_extrapolation(fit_func, x_data, n_extrapolation_steps, step_size, model_params)
                
                if extrap_error:
                    extrapolation_output = f"❌ **Extrapolation Error**: {extrap_error}"
                else:
                    extrapolation_output = format_extrapolation_output(x_extrap, y_extrap, n_extrapolation_steps)
                    
                    # Add extrapolation points to plot
                    if x_extrap is not None and y_extrap is not None:
                        fig.add_trace(go.Scatter(
                            x=x_extrap, y=y_extrap,
                            mode='markers+lines',
                            name='Extrapolation',
                            line=dict(color='purple', dash='dot', width=2),
                            marker=dict(color='purple', symbol='diamond', size=8)
                        ))
                        
                        # Add annotations for extrapolation points
                        for i, (x, y) in enumerate(zip(x_extrap, y_extrap)):
                            fig.add_annotation(
                                x=x, y=y,
                                text=f"Point {i+1}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor="purple",
                                ax=20,
                                ay=-30
                            )
            except Exception as e:
                extrapolation_output = f"❌ **Extrapolation Error**: {str(e)}"
        else:
            extrapolation_output = "🔮 **Extrapolation**: Enable to predict future values based on the fitted model"

        # Add special points and features
        _add_special_points(fig, fit_func, x_data, y_data, special_points_show, fit_type, degree)
        
        # Add derivative calculation
        if x_for_derivative is not None and fit_type == "polynomial":
            derivative_text_output = _calculate_derivative(fig, fit_func, x_for_derivative, x_data, degree)
        elif x_for_derivative is not None and fit_type != "polynomial":
            derivative_text_output = f"ℹ️ **Derivative calculation**: Only available for polynomial fits. Current fit type: {fit_type}"

        # Add area calculation
        if show_area and area_start_x is not None and area_end_x is not None:
            area_output = _calculate_and_show_area(fig, fit_func, area_start_x, area_end_x, x_data, y_data, model_params)
        else:
            area_output = "📐 **Area Calculation**: Enable 'Show Area' and set X boundaries to calculate area under curve"

        # Configure layout
        _configure_plot_layout(fig, plot_options, xaxis_scale, yaxis_scale, custom_x_label, custom_y_label, 
                              custom_title, font_family, title_font_size, axes_font_size, legend_font_size, 
                              font_color, x_data, y_data, info_msg)

        # Combine equation and info messages
        full_message = f"**{equation_str}**\n\n{info_msg}"
        
        return fig, full_message, derivative_text_output, statistics_output, area_output, extrapolation_output, updated_slider

    except Exception as e:
        return None, f"🔴 **Unexpected Error**: {str(e)}", "", "", "", "", default_slider

def _add_special_points(fig, fit_func, x_data, y_data, special_points_show, fit_type, degree):
    """Add special points to the plot."""
    # Extrema (critical points) - only for polynomial fits
    if "Show Extrema" in special_points_show and fit_type == "polynomial" and degree > 0:
        try:
            deriv_poly = fit_func.deriv()
            extrema_x_complex = deriv_poly.roots
            extrema_x = extrema_x_complex[np.isreal(extrema_x_complex)].real
            if len(x_data) > 0:
                extrema_x = extrema_x[(extrema_x >= x_data.min()) & (extrema_x <= x_data.max())]
            if extrema_x.size > 0:
                extrema_y = fit_func(extrema_x)
                fig.add_trace(go.Scatter(
                    x=extrema_x, y=extrema_y, mode='markers', name='Extrema',
                    marker=dict(color='red', size=10, symbol='star')
                ))
        except:
            pass
    
    # X-Intercepts - only for polynomial fits
    if "Show X-Intercepts" in special_points_show and fit_type == "polynomial" and degree > 0:
        try:
            x_intercepts_complex = fit_func.roots
            x_intercepts = x_intercepts_complex[np.isreal(x_intercepts_complex)].real
            if len(x_data) > 0:
                x_intercepts = x_intercepts[(x_intercepts >= x_data.min()) & (x_intercepts <= x_data.max())]
            if x_intercepts.size > 0:
                y_intercepts = np.zeros_like(x_intercepts)
                fig.add_trace(go.Scatter(
                    x=x_intercepts, y=y_intercepts, mode='markers', name='X-Intercepts',
                    marker=dict(color='green', size=10, symbol='x')
                ))
        except:
            pass
    
    # Y-Intercept
    if "Show Y-Intercept" in special_points_show:
        try:
            y_intercept = fit_func(0)
            if not (np.isnan(y_intercept) or np.isinf(y_intercept)):
                fig.add_trace(go.Scatter(
                    x=[0], y=[y_intercept], mode='markers', name='Y-Intercept',
                    marker=dict(color='orange', size=10, symbol='circle-open')
                ))
        except:
            pass


def _calculate_derivative(fig, fit_func, x_for_derivative, x_data, degree):
    """Calculate and display derivative at a point."""
    try:
        x_val = float(x_for_derivative)
        
        # Check if x_val is within a reasonable range
        if len(x_data) > 1:
            x_range = x_data.max() - x_data.min()
            buffer = x_range * 0.5  # Allow 50% extension beyond data range
            if not (x_data.min() - buffer <= x_val <= x_data.max() + buffer):
                return f"⚠️ Warning: X={x_val:.2f} is far from data range [{x_data.min():.2f}, {x_data.max():.2f}]"
        
        if degree > 0:
            deriv_poly = fit_func.deriv()
            slope_at_point = deriv_poly(x_val)
            
            # Check for extreme derivative values
            if abs(slope_at_point) > 1e6:
                return f"⚠️ f'({x_val:.2e}) = {slope_at_point:.2e} (very large!)"
            else:
                return f"✅ f'({x_val:.2f}) = {slope_at_point:.3f}"
        else:
            return f"✅ f'({x_val:.2f}) = 0 (constant function)"
        
    except ValueError:
        return "❌ Invalid X value for derivative calculation"
    except Exception as e:
        return f"❌ Error calculating derivative: {str(e)}"


def _calculate_and_show_area(fig, fit_func, area_start_x, area_end_x, x_data, y_data, model_params):
    """Calculate and visualize area under curve."""
    try:
        start_x = float(area_start_x)
        end_x = float(area_end_x)
        
        if start_x < end_x:
            # For area calculation, create a lambda function for non-polynomial models
            if model_params is not None:
                area_func = lambda x: fit_func(x, *model_params)
            else:
                area_func = fit_func
            
            # Calculate area
            area_value, area_error = calculate_area_under_curve(area_func, start_x, end_x)
            
            if area_error:
                return f"❌ **Area Error**: {area_error}"
            else:
                area_output = format_area_output(area_value, start_x, end_x, area_func)
                
                # Create area visualization
                if len(x_data) > 0:
                    # Create fine-grained x points for smooth area fill
                    x_area = np.linspace(start_x, end_x, 200)
                    if model_params is not None:
                        y_area = fit_func(x_area, *model_params)
                    else:
                        y_area = fit_func(x_area)
                    
                    # Add area fill
                    fig.add_trace(go.Scatter(
                        x=x_area, y=y_area,
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
                        name=f'Area: {area_value:.4f}',
                        hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Area Region<extra></extra>'
                    ))
                
                return area_output
        else:
            return "❌ **Area Error**: Start X must be less than End X"
            
    except ValueError:
        return "❌ **Area Error**: Invalid X values for area calculation"
    except Exception as e:
        return f"❌ **Area Error**: {str(e)}"


def _configure_plot_layout(fig, plot_options, xaxis_scale, yaxis_scale, custom_x_label, custom_y_label, 
                          custom_title, font_family, title_font_size, axes_font_size, legend_font_size, 
                          font_color, x_data, y_data, info_msg):
    """Configure the plot layout and styling."""
    layout_options = {}
    
    try:
        # Grid options
        if "Show Grid" in plot_options:
            layout_options['xaxis_showgrid'] = True
            layout_options['yaxis_showgrid'] = True
        else:
            layout_options['xaxis_showgrid'] = False
            layout_options['yaxis_showgrid'] = False

        # Labels with fallbacks
        if "Show Axes Labels" in plot_options:
            x_label = custom_x_label.strip() if custom_x_label and custom_x_label.strip() else "X values"
            y_label = custom_y_label.strip() if custom_y_label and custom_y_label.strip() else "Y values"
            layout_options['xaxis_title'] = x_label
            layout_options['yaxis_title'] = y_label
        
        if "Show Title" in plot_options:
            title = custom_title.strip() if custom_title and custom_title.strip() else "Data and Polynomial Fit"
            layout_options['title'] = title
        else:
            title = "Data and Polynomial Fit"  # Default title for filename

        # Axis scaling with validation
        if xaxis_scale == "log" and np.any(x_data <= 0):
            xaxis_scale = "linear"
            info_msg += "\n⚠️ Switched to linear X-axis (negative/zero values detected)"
        
        if yaxis_scale == "log" and np.any(y_data <= 0):
            yaxis_scale = "linear"
            info_msg += "\n⚠️ Switched to linear Y-axis (negative/zero values detected)"

        layout_options['xaxis_type'] = xaxis_scale
        layout_options['yaxis_type'] = yaxis_scale
        layout_options['legend_title_text'] = 'Legend'
        
        # Font customization with validation
        safe_font_family = font_family if font_family else "Arial"
        safe_title_size = max(8, min(32, title_font_size)) if title_font_size else 16
        safe_axes_size = max(8, min(24, axes_font_size)) if axes_font_size else 12
        safe_legend_size = max(8, min(20, legend_font_size)) if legend_font_size else 10
        
        layout_options['font'] = dict(
            family=safe_font_family,
            size=safe_axes_size,
            color=font_color if font_color else "#000000"
        )
        
        if "Show Title" in plot_options:
            layout_options['title_font'] = dict(
                family=safe_font_family,
                size=safe_title_size,
                color=font_color if font_color else "#000000"
            )
        
        layout_options['legend_font'] = dict(
            family=safe_font_family,
            size=safe_legend_size,
            color=font_color if font_color else "#000000"
        )
        
        fig.update_layout(**layout_options, height=500, margin=dict(l=50, r=50, b=50, t=50, pad=4))
        
    except Exception as e:
        print(f"Layout configuration warning: {str(e)}")


def update_combined_plot(
    curve_configs, plot_options, xaxis_scale, yaxis_scale, special_points_show, x_for_derivative,
    custom_x_label, custom_y_label, custom_title,
    font_family, title_font_size, axes_font_size, legend_font_size, font_color,
    area_start_x, area_end_x, show_area, n_extrapolation_steps, extrapolation_step_size, show_extrapolation,
    connect_points
):
    """Update plot with multiple datasets."""
    try:
        import gradio as gr
        derivative_text_output = ""
        statistics_output = ""
        area_output = ""
        extrapolation_output = ""
        
        if not curve_configs:
            return None, "❌ No datasets configured", "", "", "", ""
        
        visible_curves = [config for config in curve_configs if config.get('visible', True)]
        if not visible_curves:
            return None, "❌ No visible datasets", "", "", "", ""
        
        fig = go.Figure() #Пустая фигура
        all_x_data = []  #Пустые списки
        all_y_data = []
        curve_data = []
        

        # Process each visible curve
        for i, config in enumerate(visible_curves):            # Проходимся по видимым прямым
            x_data, y_data, z_data = [], [], None
            current_curve_info_msg = ""
            
            # Get data for this curve
            if config.get('file_df') is not None and config.get('x_col_name') and config.get('y_col_name'): # Use .get for safety
                try:
                    file_df_actual = config['file_df'] # Already a DataFrame
                    x_data_series = pd.to_numeric(file_df_actual[config['x_col_name']], errors='coerce')             # Если данные есть и колонки не пусты
                    y_data_series = pd.to_numeric(file_df_actual[config['y_col_name']], errors='coerce')
                    
                    # Handle Z column if provided
                    z_data_series = None
                    if config.get('z_col_name') and config['z_col_name'] in file_df_actual.columns:                    # Если Z включён
                        z_data_series = pd.to_numeric(file_df_actual[config['z_col_name']], errors='coerce')

                    # Create combined dataframe and remove rows with NaN
                    if z_data_series is not None:
                        combined_df = pd.DataFrame({
                            'x': x_data_series, 
                            'y': y_data_series,
                            'z': z_data_series
                        }).dropna()
                        z_data = combined_df['z'].to_numpy()
                    else:
                        combined_df = pd.DataFrame({
                            'x': x_data_series, 
                            'y': y_data_series
                        }).dropna()
                        z_data = None # Ensure z_data is None if not present or not valid                           # соеденили всё в одно datdframe
                    
                    x_data = combined_df['x'].to_numpy()
                    y_data = combined_df['y'].to_numpy()
                except Exception as e:
                    # If file processing fails for a curve, skip it and add a message
                    statistics_output += f"\n⚠️ Error processing file data for {config.get('name', 'Unnamed Curve')}: {str(e)}"
                    continue
            else:
                # Parse manual input data
                x_parsed, x_error = parse_input_data(config.get('x_text', ''))
                y_parsed, y_error = parse_input_data(config.get('y_text', ''))              # Получаем чисолвые массивы
                z_parsed, z_error = parse_input_data(config.get('z_text', ''))
                
                if not x_error and not y_error and len(x_parsed) > 0 and len(y_parsed) > 0 and len(x_parsed) == len(y_parsed):
                    x_data, y_data = x_parsed, y_parsed
                    
                    # Handle Z data for manual input
                    if not z_error and len(z_parsed) > 0:                         # Для Z
                        if len(z_parsed) == len(x_data):
                            z_data = z_parsed
                        # If Z data length doesn't match, ignore it but don't fail
            
            if len(x_data) == 0 or len(y_data) == 0:
                statistics_output += f"\n⚠️ No valid data for {config.get('name', 'Unnamed Curve')}."
                continue
            
            # Check for 3D surface plotting for this specific curve if force_3d is true
            # This logic is primarily for when a single dataset is plotted via update_combined_plot
            if len(visible_curves) == 1 and config.get('force_3d', False) and z_data is not None:
                if not _is_suitable_for_3d_surface(x_data, y_data, z_data):
                    current_curve_info_msg = "\n⚠️ **3D Plot Warning**: Data may not be ideal for 3D surface, but attempting due to explicit request."
                try:
                    surface_fig = _create_3d_surface_plot(
                        x_data, y_data, z_data,
                        None, None, # No error bars in 3D view
                        config.get('data_color', '#1f77b4'),
                        custom_x_label or 'X',
                        custom_y_label or 'Y',
                        f"{config.get('name', 'Plot')} - 3D Surface"
                    )
                    # Apply layout customizations for 3D
                    _configure_plot_layout(surface_fig, plot_options, xaxis_scale, yaxis_scale, custom_x_label, custom_y_label, 
                                          f"{config.get('name', 'Plot')} - 3D Surface", font_family, title_font_size, axes_font_size, legend_font_size, 
                                          font_color, x_data, y_data, current_curve_info_msg)  #настройка визуального оформления графика -> 
                    
                    return (
                        surface_fig,
                        f"✅ **3D Surface Plot**: {config.get('name', 'Plot')}{current_curve_info_msg}",
                        "", # derivative_text_output
                        f"📊 **Dataset**: {config.get('name', 'Plot')}\n• Points: {len(x_data)}\n• Z-data range: [{np.min(z_data):.4f}, {np.max(z_data):.4f}]" if z_data is not None and len(z_data) > 0 else "", # statistics_output
                        "", # area_output
                        ""  # extrapolation_output
                    )
                except Exception as e:
                    # If 3D plot fails, fall back to 2D combined plot
                    statistics_output += f"\n⚠️ Failed to create 3D plot for {config.get('name', 'Unnamed Curve')}: {str(e)}. Showing 2D plot instead."
                    # Continue to add to 2D plot

            curve_data.append({
                'name': config.get('name', f'Curve {i+1}'), # Use .get for safety
                'x_data': x_data,
                'y_data': y_data,
                'z_data': z_data,
                'config': config
            })
            
            all_x_data.extend(x_data)
            all_y_data.extend(y_data)
            
            # Add data points if enabled
            if config.get('show_data', True):                    # Не получился 3D график => строим 2D
                scatter_params = {
                    'x': x_data,
                    'y': y_data,
                    'mode': 'markers',
                    'name': f"{config.get('name', f'Curve {i+1}')} - Data",
                    'marker': dict(color=config.get('data_color', '#1f77b4'), symbol=config.get('data_marker', 'circle'), size=8)
                }
                
                if connect_points:  # Используем переданный параметр
                    fig.add_trace(go.Scatter(
                         x=x_data,
                         y=y_data,
                         mode='lines',
                         name=f"{config.get('name', f'Curve {i+1}')} - Lines",
                         line=dict(color=config.get('data_color', '#1f77b4'), width=1, dash='solid'),
                         opacity=0.7,
                         hoverinfo='skip'
                         ))
                # Enhanced visualization if Z-data is available (for 2D plots)
                if z_data is not None:
                    scatter_params['marker']['color'] = z_data
                    scatter_params['marker']['colorscale'] = 'Viridis'
                    scatter_params['marker']['showscale'] = True
                    scatter_params['marker']['colorbar'] = dict(title=f"Z ({config.get('name', f'Curve {i+1}')})")    # Раскрашиваем точки под Z
                    scatter_params['name'] = f"{config.get('name', f'Curve {i+1}')} - Data (Z-color)"
                
                # Add error bars if configured
                if config.get('show_error_bars', False):
                    x_errors_text = config.get('x_errors_text', '')
                    y_errors_text = config.get('y_errors_text', '')
                    x_errors, _ = parse_input_data(x_errors_text) if x_errors_text.strip() else ([], None)   #парсинг погрешностей
                    y_errors, _ = parse_input_data(y_errors_text) if y_errors_text.strip() else ([], None)
                    
                    error_added = False                                                  # выключены ерроры
                    if x_errors is not None and len(x_errors) == len(x_data):
                        scatter_params['error_x'] = dict(
                            type='data', array=x_errors, color=config.get('data_color', '#1f77b4'), thickness=1.5, width=3
                        )
                        error_added = True
                    if y_errors is not None and len(y_errors) == len(y_data):
                        scatter_params['error_y'] = dict(
                            type='data', array=y_errors, color=config.get('data_color', '#1f77b4'), thickness=1.5, width=3
                        )
                        error_added = True
                    if error_added:
                         scatter_params['name'] += ' (errors)'
                
                fig.add_trace(go.Scatter(**scatter_params))    # Точечный график
            
            # Add fitted curve if enabled
            if config.get('show_fit', True) and config.get('fit_type') == 'polynomial': # Only polynomial for combined plot for now
                try:
                    degree = min(config.get('degree', 1), len(x_data) - 1 if len(x_data) > 1 else 0)
                    if degree >= 0 and len(x_data) > 0: # Ensure degree is valid and data exists
                        coeffs = np.polyfit(x_data, y_data, degree)
                        fit_func = np.poly1d(coeffs)
                        
                        # Generate fit line only if there's more than one point to define a range
                        if len(x_data) > 1:
                            x_fit = np.linspace(np.min(x_data), np.max(x_data), 200)
                        else: # Single point, fit line is just the point itself
                            x_fit = np.array([x_data[0]])
                            
                        y_fit = fit_func(x_fit)


                        
                        # Check for numerical issues
                        if not (np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit))):
                            fig.add_trace(go.Scatter(
                                x=x_fit,
                                y=y_fit,
                                mode='lines',
                                name=f"{config.get('name', f'Curve {i+1}')} - Fit (deg {degree})",
                                line=dict(color=config.get('fit_color', '#ff7f0e'), dash=config.get('fit_line_style', 'solid'), width=2)
                            ))

                            
                            # Calculate R-squared
                            if len(x_data) > 1:
                                y_pred = fit_func(x_data)
                                ss_res = np.sum((y_data - y_pred) ** 2)
                                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res == 0 else 0) # Handle ss_tot = 0 case
                                
                                # Add R-squared to info text
                                statistics_output += f"\n\n📈 **{config.get('name', f'Curve {i+1}')} Fit Statistics**:\n• Degree: {degree}\n• R² = {r_squared:.4f}"
                            elif len(x_data) == 1:
                                statistics_output += f"\n\n📈 **{config.get('name', f'Curve {i+1}')} Fit Statistics**:\n• Single data point. Fit is constant: y = {y_data[0]:.4f}"
                except Exception as e:
                    statistics_output += f"\n⚠️ Error fitting polynomial for {config.get('name', 'Unnamed Curve')}: {str(e)}"
                    pass  # Ignore fitting errors for individual curves
        
        



        if not curve_data: # If all curves were skipped due to errors or no data
            return error_msgt, "❌ No valid data found in any visible dataset.Check the number of X and Y values. {error_msgt}", "", statistics_output, "", ""

         # Calculate combined statistics
        if all_x_data and all_y_data:
            all_x_array = np.array(all_x_data)
            all_y_array = np.array(all_y_data)
            
            stats, _ = calculate_statistics(all_x_array, all_y_array)
            if stats:
                statistics_output = f"📊 **Combined Statistics** ({len(curve_data)} datasets, {stats['n_points']} total points)\n\n"
                statistics_output += format_statistics_output(stats)
                
                # Add individual dataset info
                statistics_output += "\n\n📋 **Individual Datasets:**\n"
                for curve in curve_data:
                    config = curve['config']
                    dataset_stats, _ = calculate_statistics(curve['x_data'], curve['y_data'])
                    if dataset_stats:
                        r_sq_info = f" (R² = {config.get('r_squared', 0):.4f})" if config.get('r_squared') else ""
                        z_info = " [with Z-data]" if curve['z_data'] is not None else ""
                        statistics_output += f"• **{config['name']}**: {dataset_stats['n_points']} points{r_sq_info}{z_info}\n"
                        if 'equation' in config:
                            statistics_output += f"  Equation: {config['equation']}\n"
                        elif config.get('fit_type') == 'polynomial':
                            # Если уравнение не сохранено, но это полином, вычислите его
                            degree = min(config.get('degree', 1), len(curve['x_data']) - 1)
                            coeffs = np.polyfit(curve['x_data'], curve['y_data'], degree)
                            equation = format_polynomial(coeffs, degree)
                            statistics_output += f"  Equation: {equation}\n"
                        if config.get('equation'):
                            statistics_output += f"  {config['equation']}\n"
                        
                        # Add Z statistics if available
                        if curve['z_data'] is not None:
                            z_stats = f"  Z-data: Mean = {np.mean(curve['z_data']):.4f}, Range = [{np.min(curve['z_data']):.4f}, {np.max(curve['z_data']):.4f}]\n"
                            statistics_output += z_stats

        # Add extrapolation for first visible curve
        if show_extrapolation and curve_data:
            first_curve = curve_data[0]
            try:
                degree = min(first_curve['config']['degree'], len(first_curve['x_data']) - 1)
                coeffs = np.polyfit(first_curve['x_data'], first_curve['y_data'], degree)
                fit_func = np.poly1d(coeffs)
                
                step_size = extrapolation_step_size if extrapolation_step_size and extrapolation_step_size > 0 else None
                x_extrap, y_extrap, extrap_error = calculate_extrapolation(
                    fit_func, first_curve['x_data'], n_extrapolation_steps, step_size
                )
                
                if not extrap_error and x_extrap is not None:
                    extrapolation_output = format_extrapolation_output(x_extrap, y_extrap, n_extrapolation_steps)
                    extrapolation_output = f"🔮 **Extrapolation** (based on {first_curve['name']})\n\n" + extrapolation_output
                    
                    # Add extrapolation to plot
                    fig.add_trace(go.Scatter(
                        x=x_extrap, y=y_extrap, mode='markers+lines',
                        name=f'Extrapolation ({first_curve["name"]})',
                        marker=dict(color='purple', size=6, symbol='diamond'),
                        line=dict(color='purple', dash='dot', width=2),
                        opacity=0.8
                    ))
            except:
                pass
        
        if not extrapolation_output:
            extrapolation_output = "🔮 **Extrapolation**: Enable to predict future values (based on first dataset)"
        
        # Add area calculation for first curve
        if show_area and area_start_x is not None and area_end_x is not None and curve_data:
            first_curve = curve_data[0]
            try:
                start_x, end_x = float(area_start_x), float(area_end_x)
                if start_x < end_x:
                    degree = min(first_curve['config']['degree'], len(first_curve['x_data']) - 1)
                    coeffs = np.polyfit(first_curve['x_data'], first_curve['y_data'], degree)
                    fit_func = np.poly1d(coeffs)
                    
                    area_value, area_error = calculate_area_under_curve(fit_func, start_x, end_x)
                    if not area_error:
                        area_output = f"📐 **Area Under Curve** (based on {first_curve['name']})\n\n"
                        area_output += format_area_output(area_value, start_x, end_x, fit_func)
                        
                        # Add area visualization
                        x_area = np.linspace(start_x, end_x, 200)
                        y_area = fit_func(x_area)
                        
                        fig.add_trace(go.Scatter(
                            x=x_area, y=y_area, fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
                            name=f'Area: {area_value:.4f}',
                            hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Area Region<extra></extra>'
                        ))
            except:
                pass
        
        if not area_output:
            area_output = "📐 **Area Calculation**: Enable to calculate area under curve (based on first dataset)"

        # --- Finalize Layout and Return ---
        # Common layout adjustments for combined plots
        _configure_plot_layout(fig, plot_options, xaxis_scale, yaxis_scale, custom_x_label, custom_y_label, 
                              custom_title, font_family, title_font_size, axes_font_size, legend_font_size, 
                              font_color, all_x_data, all_y_data, "")
        
        # Add a general message if some curves had issues but others plotted
        if "⚠️" in statistics_output:
            final_message = "✅ **Combined Plot**: Plotted available valid datasets. Some datasets had issues (see details below)."
        else:
            final_message = "✅ **Combined Plot**: Multiple datasets plotted successfully."


        if x_for_derivative is not None and curve_data:
            first_curve = curve_data[0]
            if first_curve['config'].get('fit_type') == 'polynomial':
                try:
                    degree = min(first_curve['config'].get('degree', 1), 
                                len(first_curve['x_data']) - 1)
                    coeffs = np.polyfit(first_curve['x_data'], first_curve['y_data'], degree)
                    fit_func = np.poly1d(coeffs)
                    derivative_text_output = _calculate_derivative(
                        fig, fit_func, x_for_derivative, 
                        first_curve['x_data'], degree
                    )
                    # Добавим значение функции в точке
                    y_at_point = fit_func(float(x_for_derivative))
                    derivative_text_output += f"\n• f({x_for_derivative}) = {y_at_point:.3f}"
                except Exception as e:
                    derivative_text_output = f"❌ Error: {str(e)}"
            else:
                derivative_text_output = f"ℹ️ Derivatives only for polynomial fits (current: {first_curve['config'].get('fit_type')})"


        return fig, final_message, derivative_text_output, statistics_output, area_output, extrapolation_output
    
    except Exception as e:
        return None, f"🔴 **Unexpected Error in Combined Plot**: {str(e)}", "", "", "", ""

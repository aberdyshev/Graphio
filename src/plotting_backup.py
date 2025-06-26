import gradio as gr
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime
from scipy import integrate
from scipy.optimize import curve_fit
import warnings
import pandas as pd
from plotly.subplots import make_subplots

# Helper function to parse input strings into numpy arrays
def parse_input_data(text_data):
    """Parses comma, space, or newline separated numbers into a numpy array."""
    if not text_data.strip():
        return np.array([]), None
    
    # Replace commas and newlines with spaces, then split
    numbers_str = re.split(r'[,\s\n]+', text_data.strip())
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

# Helper function to format polynomial equation
def format_polynomial(coeffs, degree):
    """Formats numpy polynomial coefficients into a readable string."""
    terms = []
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if abs(coeff) < 1e-6:  # Skip terms with very small coefficients
             continue

        coeff_str = f"{coeff:.3f}"
        if i > 0:
            coeff_str = f"+ {abs(coeff):.3f}" if coeff > 0 else f"- {abs(coeff):.3f}"

        if power == 0:
            terms.append(f"{coeff_str}")
        elif power == 1:
            terms.append(f"{coeff_str}x")
        else:
            terms.append(f"{coeff_str}x^{power}")

    if not terms:
        return "f(x) = 0"

    # Adjust the first term's sign if it starts with '+'
    equation = " ".join(terms)
    if equation.startswith("+ "):
         equation = equation[2:]
    elif equation.startswith("- "):
         # Keep the minus sign but remove the space for the first term
         equation = "-" + equation[2:]


    return f"f(x) = {equation}"

# Helper function to calculate statistics
def calculate_statistics(x_data, y_data):
    """Calculate comprehensive statistics for the data."""
    try:
        stats = {}
        
        # Basic statistics for X data
        stats['x_mean'] = np.mean(x_data)
        stats['x_std'] = np.std(x_data, ddof=1) if len(x_data) > 1 else 0.0
        stats['x_min'] = np.min(x_data)
        stats['x_max'] = np.max(x_data)
        stats['x_range'] = stats['x_max'] - stats['x_min']
        
        # Basic statistics for Y data
        stats['y_mean'] = np.mean(y_data)
        stats['y_std'] = np.std(y_data, ddof=1) if len(y_data) > 1 else 0.0
        stats['y_min'] = np.min(y_data)
        stats['y_max'] = np.max(y_data)
        stats['y_range'] = stats['y_max'] - stats['y_min']
        
        # Correlation coefficient
        if len(x_data) > 1 and len(y_data) > 1:
            correlation_matrix = np.corrcoef(x_data, y_data)
            stats['correlation'] = correlation_matrix[0, 1]
            
            # Interpret correlation strength
            abs_corr = abs(stats['correlation'])
            if abs_corr >= 0.9:
                stats['corr_strength'] = "Very Strong"
            elif abs_corr >= 0.7:
                stats['corr_strength'] = "Strong"
            elif abs_corr >= 0.5:
                stats['corr_strength'] = "Moderate"
            elif abs_corr >= 0.3:
                stats['corr_strength'] = "Weak"
            else:
                stats['corr_strength'] = "Very Weak"
                
            # Correlation direction
            stats['corr_direction'] = "Positive" if stats['correlation'] >= 0 else "Negative"
        else:
            stats['correlation'] = 0.0
            stats['corr_strength'] = "N/A"
            stats['corr_direction'] = "N/A"
        
        # Data count
        stats['n_points'] = len(x_data)
        
        return stats, None
        
    except Exception as e:
        return None, f"Error calculating statistics: {str(e)}"

def format_statistics_output(stats):
    """Format statistics into a beautiful display string."""
    if stats is None:
        return "❌ Unable to calculate statistics"
    
    # Format the statistics with emojis and clear structure
    output = f"""📊 **Statistical Summary** ({stats['n_points']} data points)

**📈 X Data Statistics:**
• Mean (μ): {stats['x_mean']:.4f}
• Std Dev (σ): {stats['x_std']:.4f}
• Range: [{stats['x_min']:.4f}, {stats['x_max']:.4f}]
• Span: {stats['x_range']:.4f}

**📉 Y Data Statistics:**
• Mean (μ): {stats['y_mean']:.4f}
• Std Dev (σ): {stats['y_std']:.4f}
• Range: [{stats['y_min']:.4f}, {stats['y_max']:.4f}]
• Span: {stats['y_range']:.4f}

**🔗 Correlation Analysis:**
• Coefficient (r): {stats['correlation']:.4f}
• Strength: {stats['corr_strength']}
• Direction: {stats['corr_direction']}"""

    # Add interpretation
    if stats['correlation'] != 0 and stats['corr_strength'] != "N/A":
        if abs(stats['correlation']) >= 0.7:
            output += f"\n• 💡 *{stats['corr_strength']} {stats['corr_direction'].lower()} relationship detected*"
        elif abs(stats['correlation']) >= 0.3:
            output += f"\n• 🤔 *{stats['corr_strength']} {stats['corr_direction'].lower()} relationship*"
        else:
            output += f"\n• 📝 *Little to no linear relationship*"

    return output

# Helper function to calculate area under curve
def calculate_area_under_curve(poly_func, x_start, x_end, method='quad'):
    """Calculate area under polynomial curve between x_start and x_end."""
    try:
        if x_start >= x_end:
            return None, "❌ Start X must be less than End X"
        
        if method == 'quad':
            # Numerical integration using scipy
            area, error = integrate.quad(poly_func, x_start, x_end)
            return area, None
        elif method == 'trapz':
            # Trapezoidal rule
            x_points = np.linspace(x_start, x_end, 1000)
            y_points = poly_func(x_points)
            area = np.trapz(y_points, x_points)
            return area, None
        else:
            return None, "❌ Unknown integration method"
            
    except Exception as e:
        return None, f"❌ Error calculating area: {str(e)}"

def format_area_output(area, x_start, x_end, poly_func=None):
    """Format area calculation results into a beautiful display string."""
    if area is None:
        return "❌ Unable to calculate area"
    
    # Calculate some additional info
    width = x_end - x_start
    avg_height = area / width if width != 0 else 0
    
    output = f"""📐 **Area Under Curve**

**📏 Integration Bounds:**
• Start X: {x_start:.4f}
• End X: {x_end:.4f}
• Width: {width:.4f}

**📊 Area Calculation:**
• Area: {area:.6f}
• Average Height: {avg_height:.6f}

**🔍 Interpretation:**
"""
    
    if area > 0:
        output += "• ✅ Positive area (curve above X-axis)"
    elif area < 0:
        output += "• ⚠️ Negative area (curve below X-axis)"
    else:
        output += "• ➖ Zero area (curve on X-axis)"
    
    if abs(area) >= 1000:
        output += f"\n• 📈 Large area magnitude: {area:.2e}"
    elif abs(area) < 0.001:
        output += f"\n• 🔬 Small area magnitude: {area:.2e}"
    
    return output

# Helper functions for alternative fitting models
def exponential_func(x, a, b, c):
    """Exponential function: f(x) = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def logarithmic_func(x, a, b, c):
    """Logarithmic function: f(x) = a * ln(b * x) + c"""
    return a * np.log(b * x) + c

def power_func(x, a, b):
    """Power function: f(x) = a / (x^b)"""
    return a / (x ** b)

def fit_alternative_model(x_data, y_data, model_type):
    """Fit alternative models to data."""
    try:
        if model_type == "exponential":
            # Initial guess for exponential
            popt, pcov = curve_fit(exponential_func, x_data, y_data, 
                                 p0=[1, 0.1, 0], maxfev=5000)
            return exponential_func, popt, pcov, f"f(x) = {popt[0]:.3f} * exp({popt[1]:.3f} * x) + {popt[2]:.3f}"
            
        elif model_type == "logarithmic":
            # Check for positive x values
            if np.any(x_data <= 0):
                return None, None, None, "❌ Logarithmic fit requires positive X values"
            
            popt, pcov = curve_fit(logarithmic_func, x_data, y_data, 
                                 p0=[1, 1, 0], maxfev=5000)
            return logarithmic_func, popt, pcov, f"f(x) = {popt[0]:.3f} * ln({popt[1]:.3f} * x) + {popt[2]:.3f}"
            
        elif model_type == "power":
            # Check for positive x values and avoid division by zero
            if np.any(x_data <= 0):
                return None, None, None, "❌ Power fit requires positive X values"
                
            popt, pcov = curve_fit(power_func, x_data, y_data, 
                                 p0=[1, 1], maxfev=5000)
            return power_func, popt, pcov, f"f(x) = {popt[0]:.3f} / (x^{popt[1]:.3f})"
            
        else:
            return None, None, None, "❌ Unknown model type"
            
    except Exception as e:
        return None, None, None, f"❌ Fitting failed: {str(e)}"

def calculate_extrapolation(func, x_data, n_steps, step_size=None, model_params=None):
    """Calculate extrapolated values."""
    try:
        if step_size is None:
            # Auto-calculate step size from data
            if len(x_data) > 1:
                step_size = (x_data.max() - x_data.min()) / (len(x_data) - 1)
            else:
                step_size = 1.0
        
        # Generate extrapolated x values
        x_max = x_data.max()
        x_extrap = np.array([x_max + i * step_size for i in range(1, n_steps + 1)])
        
        # Calculate extrapolated y values
        if model_params is not None:
            # For alternative models with parameters
            y_extrap = func(x_extrap, *model_params)
        else:
            # For polynomial models
            y_extrap = func(x_extrap)
            
        return x_extrap, y_extrap, None
        
    except Exception as e:
        return None, None, f"❌ Extrapolation error: {str(e)}"

def format_extrapolation_output(x_extrap, y_extrap, n_steps):
    """Format extrapolation results into a beautiful display string."""
    if x_extrap is None or y_extrap is None:
        return "❌ Unable to calculate extrapolation"
    
    output = f"""🔮 **Extrapolation Results** ({n_steps} steps ahead)

**📈 Predicted Values:**
"""
    
    for i, (x, y) in enumerate(zip(x_extrap, y_extrap)):
        output += f"• Step {i+1}: X = {x:.4f}, Y = {y:.4f}\n"
        if i >= 4:  # Show only first 5 values to save space
            remaining = len(x_extrap) - 5
            if remaining > 0:
                output += f"• ... and {remaining} more values\n"
            break
    
    # Add trend analysis
    if len(y_extrap) > 1:
        trend = np.diff(y_extrap)
        avg_change = np.mean(trend)
        
        output += f"\n**📊 Trend Analysis:**\n"
        output += f"• Average change per step: {avg_change:.4f}\n"
        
        if avg_change > 0:
            output += "• 📈 **Increasing trend** - values are rising\n"
        elif avg_change < 0:
            output += "• 📉 **Decreasing trend** - values are falling\n"
        else:
            output += "• ➖ **Stable trend** - values are constant\n"
            
        # Check for acceleration
        if len(trend) > 1:
            acceleration = np.diff(trend)
            avg_accel = np.mean(acceleration)
            if abs(avg_accel) > abs(avg_change) * 0.1:  # Significant acceleration
                if avg_accel > 0:
                    output += "• 🚀 **Accelerating growth** detected\n"
                else:
                    output += "• 🛑 **Decelerating growth** detected\n"
    
    return output

# Function to handle file uploads and changes
def handle_file_change(uploaded_file):
    """Handle file upload and extract column names for selection."""
    if uploaded_file is not None:
        try:
            file_path = uploaded_file.name
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return (
                    gr.Dropdown(choices=[], value=None),  # X column
                    gr.Dropdown(choices=[], value=None),  # Y column
                    gr.Dropdown(choices=[], value=None),  # Z column
                    None,  # No DataFrame
                    gr.Textbox(value=""),  # Clear x_input
                    gr.Textbox(value=""),  # Clear y_input
                    "❌ Unsupported file type. Please upload CSV or XLSX files."
                )

            if df.empty:
                return (
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    None,
                    gr.Textbox(value=""), 
                    gr.Textbox(value=""),
                    "⚠️ Uploaded file is empty."
                )

            cols = df.columns.tolist()
            success_msg = f"✅ File '{uploaded_file.name}' loaded successfully!\n📊 Found {len(df)} rows and {len(cols)} columns.\n🔧 Select X and Y columns to proceed."
            
            return (
                gr.Dropdown(choices=cols, value=None, label="Select X Column"),
                gr.Dropdown(choices=cols, value=None, label="Select Y Column"), 
                gr.Dropdown(choices=cols, value=None, label="Select Z Column (optional)"),
                df,  # Store DataFrame
                gr.Textbox(value=""),  # Clear manual inputs
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

# Main function to generate plot, fit polynomial, and prepare download
def update_plot_and_fit(
    x_text, y_text, degree, plot_options,
    xaxis_scale, yaxis_scale, special_points_show, x_for_derivative,
    data_color, fit_color, data_marker, fit_line_style,
    custom_x_label, custom_y_label, custom_title,
    font_family, title_font_size, axes_font_size, legend_font_size, font_color,
    area_start_x, area_end_x, show_area,
    fit_type, n_extrapolation_steps, extrapolation_step_size, show_extrapolation,
    x_errors_text, y_errors_text, show_error_bars,
    file_df, x_col_name, y_col_name, z_col_name  # New file parameters
):
    """
    Enhanced function with file upload, column selection, and Z-component support.
    """
    try:
        derivative_text_output = ""
        statistics_output = ""
        area_output = ""
        extrapolation_output = ""
        info_msg = ""
        
        # Create default slider for error cases
        default_slider = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")

        x_data = np.array([])
        y_data = np.array([])
        z_data = None
        data_source_message = ""

        # --- Data Source Determination ---
        if file_df is not None and x_col_name and y_col_name:
            # Using file data
            try:
                if x_col_name not in file_df.columns:
                    return None, f"🔴 **File Error**: X column '{x_col_name}' not found in file.", "", "", "", "", default_slider
                if y_col_name not in file_df.columns:
                    return None, f"🔴 **File Error**: Y column '{y_col_name}' not found in file.", "", "", "", "", default_slider

                # Convert to numeric, handling errors
                x_data_series = pd.to_numeric(file_df[x_col_name], errors='coerce')
                y_data_series = pd.to_numeric(file_df[y_col_name], errors='coerce')
                
                # Handle Z column if provided
                z_data_series = None
                if z_col_name and z_col_name in file_df.columns:
                    z_data_series = pd.to_numeric(file_df[z_col_name], errors='coerce')

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
                
                original_rows = len(file_df)
                valid_rows = len(combined_df)

                if valid_rows == 0:
                    col_list = f"'{x_col_name}', '{y_col_name}'"
                    if z_col_name:
                        col_list += f", '{z_col_name}'"
                    return None, f"🔴 **File Error**: No valid numeric data pairs found in columns {col_list}.", "", "", "", "", default_slider
                
                x_data = combined_df['x'].to_numpy()
                y_data = combined_df['y'].to_numpy()
                
                # Create data source message
                cols_used = f"X='{x_col_name}', Y='{y_col_name}'"
                if z_col_name and z_data is not None:
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
                return None, f"🔴 **File Processing Error**: {str(e)}", "", "", statistics_output, "", updated_slider
                
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
            return None, "📝 **No Data**: Please provide X data via manual input or file upload.", "", "", "", "", default_slider
        if len(y_data) == 0:
            return None, "📝 **No Data**: Please provide Y data via manual input or file upload.", "", "", "", "", default_slider
        
        if len(x_data) != len(y_data):
            return None, f"⚖️ **Length Mismatch**: X({len(x_data)}) ≠ Y({len(y_data)}) data points.", "", "", "", "", default_slider

        # Calculate statistics
        stats, stats_error = calculate_statistics(x_data, y_data)
        if stats_error:
            statistics_output = f"❌ **Statistics Error**: {stats_error}"
        else:
            statistics_output = format_statistics_output(stats)
        
        # Prepend data source info
        if data_source_message:
            statistics_output = f"{data_source_message}\n\n{statistics_output}"

        # Add Z-data info if available
        if z_data is not None:
            z_stats = f"\n\n📊 **Z Data Statistics:**\n• Mean: {np.mean(z_data):.4f}\n• Std Dev: {np.std(z_data, ddof=1):.4f}\n• Range: [{np.min(z_data):.4f}, {np.max(z_data):.4f}]"
            statistics_output += z_stats
        def validate_data_length(x_data, y_data):

            error_msg = ""
    
            # Определяем длины массивов
            x_len = len(x_data)
            y_len = len(y_data)
    
            # Проверяем соответствие размеров
            if x_len != y_len:
                error_msg = "❌ Ошибка: количество значений не совпадает!\n"
        
                # Определяем, каких данных не хватает
                if x_len > y_len:
                    missing_count = x_len - y_len
                    error_msg += f" - Не хватает {missing_count} значений Y (имеется {y_len}, требуется {x_len})"
                else:
                    missing_count = y_len - x_len
                    error_msg += f" - Не хватает {missing_count} значений X (имеется {x_len}, требуется {y_len})"
            
                    return error_msg



        error_message = validate_data_length(x_data, y_data)
        if error_message:
            print(error_message)
        else:
            print("Данные корректны, можно строить график")

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
                # Polynomial fitting (existing code)
                coeffs = np.polyfit(x_data, y_data, degree)
                
                if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
                    return None, "⚠️ **Numerical Issue**: The polynomial fit resulted in invalid coefficients.\n*Try reducing the polynomial degree or checking your data for outliers.*", "", "", statistics_output, "", updated_slider
                
                fit_func = np.poly1d(coeffs)
                equation_str = format_polynomial(coeffs, degree)
                model_params = None
                
                # Calculate R-squared
                if len(x_data) > 1:
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
                    
            else:
                # Alternative model fitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_func, model_params, pcov, equation_str = fit_alternative_model(x_data, y_data, fit_type)
                
                if fit_func is None:
                    return None, f"🔴 **{fit_type.title()} Fitting Error**: {equation_str}", "", "", statistics_output, "", updated_slider
                
                # Calculate R-squared for alternative models
                try:
                    y_pred = fit_func(x_data, *model_params)
                    ss_res = np.sum((y_data - y_pred) ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    if r_squared >= 0.9:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n🎯 **Excellent Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.7:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n👍 **Good Fit**: R² = {r_squared:.4f}"
                    elif r_squared >= 0.5:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n👌 **Moderate Fit**: R² = {r_squared:.4f}"
                    else:
                        info_msg = f"✅ **{fit_type.title()} Fit**\n⚠️ **Poor Fit**: R² = {r_squared:.4f}"
                except:
                    info_msg = f"✅ **{fit_type.title()} Fit** completed"

            # Generate points for the fitted curve
            if len(x_data) > 1:
                x_range = x_data.max() - x_data.min()
                if x_range == 0:
                    return None, "⚠️ **Data Issue**: All X values are identical.\n*X values must vary to create a meaningful plot.*", "", "", statistics_output, "", updated_slider
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
            elif len(x_data) == 1:
                x_fit = np.array([x_data[0] - 1, x_data[0], x_data[0] + 1])
            else:
                x_fit = np.array([])
            
            # Calculate fitted y values
            if model_params is not None:
                y_fit = fit_func(x_fit, *model_params)
            else:
                y_fit = fit_func(x_fit)
            
            # Check for numerical issues
            if np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit)):
                return None, f"⚠️ **Fitting Warning**: The {fit_type} produces extreme values.\n*Try a different fitting method.*", "", "", statistics_output, "", updated_slider

        except Exception as e:
            return None, f"🔴 **Fitting Error**: {str(e)}\n*Try adjusting the fitting method or checking your data.*", "", "", statistics_output, "", updated_slider

        # --- Extrapolation ---
        if show_extrapolation and n_extrapolation_steps > 0:
            step_size = extrapolation_step_size if extrapolation_step_size and extrapolation_step_size > 0 else None
            x_extrap, y_extrap, extrap_error = calculate_extrapolation(fit_func, x_data, n_extrapolation_steps, step_size, model_params)
            
            if extrap_error:
                extrapolation_output = extrap_error
            else:
                extrapolation_output = format_extrapolation_output(x_extrap, y_extrap, n_extrapolation_steps)
        else:
            extrapolation_output = "🔮 **Extrapolation**: Enable 'Show Extrapolation' and set number of steps to predict future values"
            x_extrap, y_extrap = None, None

        # --- Safe Plotting with Plotly ---
        try:
            fig = go.Figure()

            # Validate colors
            try:
                # Test if colors are valid
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
            
            # Enhanced visualization if Z-data is available
            if z_data is not None:
                # Use Z-data for color mapping
                scatter_params['marker']['color'] = z_data
                scatter_params['marker']['colorscale'] = 'Viridis'
                scatter_params['marker']['showscale'] = True
                scatter_params['marker']['colorbar'] = dict(title="Z Values")
                scatter_params['name'] = f'Data (colored by {z_col_name if z_col_name else "Z"})'
            
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
            fig.add_trace(go.Scatter(
                x=x_fit, y=y_fit, mode='lines', name=f'{fit_type.title()} Fit',
                line=dict(color=fit_color, dash=fit_line_style, width=2)
            ))

            # Plot extrapolated points
            if show_extrapolation and x_extrap is not None and y_extrap is not None:
                fig.add_trace(go.Scatter(
                    x=x_extrap, y=y_extrap, mode='markers+lines', 
                    name=f'Extrapolation ({n_extrapolation_steps} steps)',
                    marker=dict(color='purple', size=6, symbol='diamond'),
                    line=dict(color='purple', dash='dot', width=2),
                    opacity=0.8
                ))
                
                # Add annotations for extrapolated points
                for i, (x, y) in enumerate(zip(x_extrap[:3], y_extrap[:3])):  # Show first 3
                    fig.add_annotation(
                        x=x, y=y, text=f"Step {i+1}",
                        showarrow=True, arrowhead=1, ax=0, ay=-20,
                        font=dict(size=9, color="purple"),
                        bordercolor="purple", borderwidth=1, bgcolor="rgba(255,255,255,0.8)"
                    )

            # --- Special Points ---
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
                except Exception:
                    # Skip extrema calculation if it fails
                    pass

            # X-Intercepts (roots) - only for polynomial fits
            if "Show X-Intercepts" in special_points_show and fit_type == "polynomial":
                try:
                    x_intercepts_complex = fit_func.roots
                    x_intercepts = x_intercepts_complex[np.isreal(x_intercepts_complex)].real
                    if len(x_data) > 0:
                        x_intercepts = x_intercepts[(x_intercepts >= x_data.min()) & (x_intercepts <= x_data.max())]
                    if x_intercepts.size > 0:
                        fig.add_trace(go.Scatter(
                            x=x_intercepts, y=np.zeros_like(x_intercepts), mode='markers', name='X-Intercepts',
                            marker=dict(color='purple', size=10, symbol='diamond')
                        ))
                except Exception:
                    # Skip x-intercepts calculation if it fails
                    pass

            # Y-Intercept - works for all function types
            if "Show Y-Intercept" in special_points_show:
                try:
                    if model_params is not None:
                        y_intercept_val = fit_func(0, *model_params)
                    else:
                        y_intercept_val = fit_func(0)
                        
                    show_y_intercept = False
                    if len(x_data) > 0:
                        if x_data.min() <= 0 <= x_data.max():
                            show_y_intercept = True
                    elif len(x_fit) > 0:
                        if x_fit.min() <= 0 <= x_fit.max():
                            show_y_intercept = True
                    
                    if show_y_intercept:
                        fig.add_trace(go.Scatter(
                            x=[0], y=[y_intercept_val], mode='markers', name='Y-Intercept',
                            marker=dict(color='orange', size=10, symbol='cross')
                        ))
                except Exception:
                    # Skip y-intercept if calculation fails (e.g., ln(0) for logarithmic)
                    pass
            
            # Derivative at a point - only for polynomial fits
            if x_for_derivative is not None and fit_type == "polynomial":
                try:
                    x_val = float(x_for_derivative)
                    
                    # Check if x_val is within a reasonable range
                    if len(x_data) > 1:
                        x_range = x_data.max() - x_data.min()
                        buffer = x_range * 0.5  # Allow 50% extension beyond data range
                        if not (x_data.min() - buffer <= x_val <= x_data.max() + buffer):
                            derivative_text_output = f"⚠️ Warning: X={x_val:.2f} is far from data range [{x_data.min():.2f}, {x_data.max():.2f}]"
                    
                    if degree > 0:
                        deriv_poly = fit_func.deriv()
                        slope_at_point = deriv_poly(x_val)
                        
                        # Check for extreme derivative values
                        if abs(slope_at_point) > 1e6:
                            derivative_text_output = f"⚠️ f'({x_val:.2e}) = {slope_at_point:.2e} (very large!)"
                        else:
                            derivative_text_output = f"✅ f'({x_val:.2f}) = {slope_at_point:.3f}"
                    else:
                        slope_at_point = 0.0
                        derivative_text_output = f"✅ f'({x_val:.2f}) = 0 (constant function)"
                    
                    y_at_point = fit_func(x_val)
                    
                    # Add derivative point and annotation
                    fig.add_trace(go.Scatter(
                        x=[x_val], y=[y_at_point], mode='markers', name=f'Point for f\'(x)',
                        marker=dict(color='cyan', size=10, symbol='circle-open', line=dict(width=2))
                    ))
                    
                    fig.add_annotation(
                        x=x_val, y=y_at_point, text=f"Slope: {slope_at_point:.2f}",
                        showarrow=True, arrowhead=1, ax=20, ay=-30,
                        font=dict(size=10, color="black"),
                        bordercolor="black", borderwidth=1, bgcolor="rgba(255,255,255,0.7)"
                    )
                    
                except ValueError:
                    derivative_text_output = "❌ Invalid X value for derivative calculation"
                except Exception as e:
                    derivative_text_output = f"❌ Error calculating derivative: {str(e)}"
            elif x_for_derivative is not None and fit_type != "polynomial":
                derivative_text_output = f"ℹ️ **Derivative calculation**: Only available for polynomial fits. Current fit type: {fit_type}"

            # --- Area Under Curve ---
            if show_area and area_start_x is not None and area_end_x is not None:
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
                            area_output = f"❌ **Area Error**: {area_error}"
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
                            
                            # Add vertical lines at boundaries
                            y_min = min(min(y_data), min(y_area)) if len(y_area) > 0 else min(y_data)
                            y_max = max(max(y_data), max(y_area)) if len(y_area) > 0 else max(y_data)
                            
                            # Start boundary line
                            fig.add_trace(go.Scatter(
                                x=[start_x, start_x], y=[y_min, y_max],
                                mode='lines',
                                line=dict(color='red', width=2, dash='dash'),
                                name=f'Start: X={start_x:.2f}',
                                showlegend=False
                            ))
                            
                            # End boundary line
                            fig.add_trace(go.Scatter(
                                x=[end_x, end_x], y=[y_min, y_max],
                                mode='lines',
                                line=dict(color='red', width=2, dash='dash'),
                                name=f'End: X={end_x:.2f}',
                                showlegend=False
                            ))
                            
                            # Add annotations for boundaries
                            fig.add_annotation(
                                x=start_x, y=y_max * 0.9,
                                text=f"Start<br>X={start_x:.2f}",
                                showarrow=True, arrowhead=2, ax=0, ay=-30,
                                font=dict(size=10, color="red"),
                                bordercolor="red", borderwidth=1, bgcolor="rgba(255,255,255,0.8)"
                            )
                            
                            fig.add_annotation(
                                x=end_x, y=y_max * 0.9,
                                text=f"End<br>X={end_x:.2f}",
                                showarrow=True, arrowhead=2, ax=0, ay=-30,
                                font=dict(size=10, color="red"),
                                bordercolor="red", borderwidth=1, bgcolor="rgba(255,255,255,0.8)"
                            )
                    else:
                        area_output = "❌ **Area Error**: Start X must be less than End X"
                        
                except ValueError:
                    area_output = "❌ **Area Error**: Invalid X values for area calculation"
                except Exception as e:
                    area_output = f"❌ **Area Error**: {str(e)}"
            else:
                area_output = "📐 **Area Calculation**: Enable 'Show Area' and set X boundaries to calculate area under curve"

            # --- Layout and Plot Options with Error Handling ---
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
                
                layout_options['xaxis_title_font'] = dict(
                    family=safe_font_family,
                    size=safe_axes_size,
                    color=font_color if font_color else "#000000"
                )
                layout_options['yaxis_title_font'] = dict(
                    family=safe_font_family,
                    size=safe_axes_size,
                    color=font_color if font_color else "#000000"
                )
                
                layout_options['legend_font'] = dict(
                    family=safe_font_family,
                    size=safe_legend_size,
                    color=font_color if font_color else "#000000"
                )
                
                fig.update_layout(**layout_options, height=500, margin=dict(l=50, r=50, b=50, t=50, pad=4))
                
                # Configure custom filename for Plotly exports
                # Clean title for filename (remove special characters)
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()
                clean_title = re.sub(r'[-\s]+', '_', clean_title)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_filename = f"{clean_title}_{timestamp}"
                
                # Update title configuration properly
                if "Show Title" in plot_options:
                    fig.update_layout(
                        title=dict(
                            text=title,
                            font=dict(
                                family=safe_font_family, 
                                size=safe_title_size, 
                                color=font_color if font_color else "#000000"
                            )
                        )
                    )
                
                # Configure Plotly with custom filename using config
                # Note: This sets a suggested filename but browsers may still override it
                plotly_config = {
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': custom_filename,
                        'height': 500,
                        'width': 700,
                        'scale': 2
                    },
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'responsive': True
                }
                
                # Store config for Gradio (this might not work directly, but we'll try)
                fig._config = plotly_config
                
            except Exception as e:
                info_msg += f"\n⚠️ Layout warning: {str(e)}"

            # Combine equation and info messages
            full_message = f"**{equation_str}**\n\n{info_msg}"
            
            return fig, full_message, derivative_text_output, statistics_output, area_output, extrapolation_output, updated_slider

        except Exception as e:
            return None, f"🔴 **Plotting Error**: {str(e)}\n*Please check your styling settings and try again.*", "", statistics_output, "", "", updated_slider

    except Exception as e:
        # Catch-all for any unexpected errors
        return None, f"🔴 **Unexpected Error**: {str(e)}\n*Please refresh the page and try again. If the problem persists, check your input data.*", "", "", "", "", gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")

# Function to update only the slider when data changes
def update_slider_only(x_text, y_text):
    """Updates only the degree slider based on data points."""
    try:
        x_data, x_error = parse_input_data(x_text)
        y_data, y_error = parse_input_data(y_text)
        
        if x_error or y_error or len(x_data) == 0 or len(y_data) == 0 or len(x_data) != len(y_data):
            return gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")
        
        max_degree = max(0, len(x_data) - 1)
        return gr.Slider(minimum=0, maximum=max_degree, step=1, value=min(3, max_degree), label="Polynomial Degree (n)")
    except Exception:
        return gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Polynomial Degree (n)")

# Curve/Dataset management functions
def create_new_curve(curve_configs, curve_names, current_curve_idx):
    """Create a new curve configuration."""
    new_idx = len(curve_configs)
    new_name = f"Dataset {new_idx + 1}"
    
    # Default configuration
    default_config = {
        'name': new_name,
        'x_text': '',
        'y_text': '',
        'z_text': '',  # Add Z text field
        'degree': 3,
        'data_color': f'#{hash(new_name) % 0xFFFFFF:06x}',  # Generate unique color
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
        'visible': True
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

def duplicate_curve(curve_configs, curve_names, current_curve_idx):
    """Duplicate the current curve configuration."""
    if not curve_configs or current_curve_idx >= len(curve_configs):
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names), "❌ No dataset to duplicate"
    
    current_config = curve_configs[current_curve_idx].copy()
    new_idx = len(curve_configs)
    new_name = f"{current_config['name']} (Copy)"
    current_config['name'] = new_name
    current_config['data_color'] = f'#{hash(new_name) % 0xFFFFFF:06x}'
    current_config['fit_color'] = f'#{(hash(new_name) + 123456) % 0xFFFFFF:06x}'
    
    curve_configs.append(current_config)
    curve_names.append(new_name)
    
    return (
        curve_configs,
        curve_names,
        new_idx,
        gr.Dropdown(choices=curve_names, value=new_name, label="Select Dataset"),
        f"✅ Duplicated dataset: {new_name}"
    )

def delete_curve(curve_configs, curve_names, current_curve_idx):
    """Delete the current curve."""
    if len(curve_configs) <= 1:
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names), "❌ Cannot delete the last dataset"
    
    if current_curve_idx >= len(curve_configs):
        return curve_configs, curve_names, current_curve_idx, gr.Dropdown(choices=curve_names), "❌ Invalid dataset selection"
    
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

def switch_curve(curve_configs, curve_names, selected_curve_name):
    """Switch to a different curve configuration."""
    if selected_curve_name not in curve_names:
        return None, "❌ Dataset not found"
    
    curve_idx = curve_names.index(selected_curve_name)
    config = curve_configs[curve_idx]
    
    # Return all the UI component updates for the selected curve
    return (
        curve_idx,
        config['x_text'],
        config['y_text'],
        config.get('z_text', ''),  # Add Z text field
        config['degree'],
        config['data_color'],
        config['fit_color'],
        config['data_marker'],
        config['fit_line_style'],
        config['fit_type'],
        config['show_fit'],
        config['show_data'],
        config['x_errors_text'],
        config['y_errors_text'],
        config['show_error_bars'],
        config['visible'],
        f"✅ Switched to: {selected_curve_name}"
    )

def save_current_curve_config(curve_configs, current_curve_idx, *config_values):
    """Save the current UI state to the curve configuration."""
    if current_curve_idx >= len(curve_configs):
        return curve_configs
    
    config_keys = [
        'x_text', 'y_text', 'z_text', 'degree', 'data_color', 'fit_color',
        'data_marker', 'fit_line_style', 'fit_type', 'show_fit', 'show_data',
        'x_errors_text', 'y_errors_text', 'show_error_bars', 'visible',
        'file_df', 'x_col_name', 'y_col_name', 'z_col_name'
    ]
    
    for i, key in enumerate(config_keys):
        if i < len(config_values):
            curve_configs[current_curve_idx][key] = config_values[i]
    
    return curve_configs

def update_combined_plot(
    curve_configs, plot_options, xaxis_scale, yaxis_scale, special_points_show, x_for_derivative,
    custom_x_label, custom_y_label, custom_title,
    font_family, title_font_size, axes_font_size, legend_font_size, font_color,
    area_start_x, area_end_x, show_area, n_extrapolation_steps, extrapolation_step_size, show_extrapolation
):
    """Create a combined plot with all visible curves."""
    try:
        derivative_text_output = ""
        statistics_output = ""
        area_output = ""
        extrapolation_output = ""
        
        if not curve_configs:
            return None, "❌ No datasets configured", "", "", "", ""
        
        visible_curves = [config for config in curve_configs if config.get('visible', True)]
        if not visible_curves:
            return None, "❌ No visible datasets", "", "", "", ""
        
        fig = go.Figure()
        all_x_data = []
        all_y_data = []
        curve_data = []
        
        # Process each visible curve
        for i, config in enumerate(visible_curves):
            x_data, y_data, z_data = [], [], None
            
            # Get data for this curve
            if config['file_df'] is not None and config['x_col_name'] and config['y_col_name']:
                try:
                    x_data_series = pd.to_numeric(config['file_df'][config['x_col_name']], errors='coerce')
                    y_data_series = pd.to_numeric(config['file_df'][config['y_col_name']], errors='coerce')
                    
                    # Handle Z column if provided
                    z_data_series = None
                    if config['z_col_name'] and config['z_col_name'] in config['file_df'].columns:
                        z_data_series = pd.to_numeric(config['file_df'][config['z_col_name']], errors='coerce')

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
                        z_data = None
                    
                    x_data = combined_df['x'].to_numpy()
                    y_data = combined_df['y'].to_numpy()
                except:
                    continue
            else:
                # Parse manual input data
                x_parsed, x_error = parse_input_data(config['x_text'])
                y_parsed, y_error = parse_input_data(config['y_text'])
                z_parsed, z_error = parse_input_data(config.get('z_text', ''))
                
                if not x_error and not y_error and len(x_parsed) > 0 and len(y_parsed) > 0 and len(x_parsed) == len(y_parsed):
                    x_data, y_data = x_parsed, y_parsed
                    
                    # Handle Z data for manual input
                    if not z_error and len(z_parsed) > 0:
                        if len(z_parsed) == len(x_data):
                            z_data = z_parsed
                        # If Z data length doesn't match, ignore it but don't fail
            
            if len(x_data) == 0 or len(y_data) == 0:
                continue
            
            curve_data.append({
                'name': config['name'],
                'x_data': x_data,
                'y_data': y_data,
                'z_data': z_data,
                'config': config
            })
            
            all_x_data.extend(x_data)
            all_y_data.extend(y_data)
            
            # Add data points if enabled
            if config.get('show_data', True):
                scatter_params = {
                    'x': x_data,
                    'y': y_data,
                    'mode': 'markers',
                    'name': f'{config["name"]} - Data',
                    'marker': dict(color=config['data_color'], symbol=config['data_marker'], size=8)
                }
                
                # Enhanced visualization if Z-data is available
                if z_data is not None:
                    scatter_params['marker']['color'] = z_data
                    scatter_params['marker']['colorscale'] = 'Viridis'
                    scatter_params['marker']['showscale'] = True
                    scatter_params['marker']['colorbar'] = dict(title=f"Z Values ({config['name']})")
                    scatter_params['name'] = f'{config["name"]} - Data (colored by Z)'
                
                # Add error bars if configured
                if config.get('show_error_bars', False):
                    x_errors, _ = parse_input_data(config.get('x_errors_text', ''))
                    y_errors, _ = parse_input_data(config.get('y_errors_text', ''))
                    
                    if len(x_errors) == len(x_data):
                        scatter_params['error_x'] = dict(
                            type='data', array=x_errors, color=config['data_color'], thickness=1.5, width=3
                        )
                    if len(y_errors) == len(y_data):
                        scatter_params['error_y'] = dict(
                            type='data', array=y_errors, color=config['data_color'], thickness=1.5, width=3
                        )
                
                fig.add_trace(go.Scatter(**scatter_params))
            
            # Add fitted curve if enabled
            if config.get('show_fit', True) and config['fit_type'] == 'polynomial':
                try:
                    degree = min(config['degree'], len(x_data) - 1)
                    if degree >= 0:
                        coeffs = np.polyfit(x_data, y_data, degree)
                        fit_func = np.poly1d(coeffs)
                        
                        x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                        y_fit = fit_func(x_fit)
                        
                        # Check for numerical issues
                        if not (np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit))):
                            fig.add_trace(go.Scatter(
                                x=x_fit,
                                y=y_fit,
                                mode='lines',
                                name=f'{config["name"]} - Fit (deg {degree})',
                                line=dict(color=config['fit_color'], dash=config['fit_line_style'], width=2)
                            ))
                            
                            # Calculate R-squared
                            if len(x_data) > 1:
                                y_pred = fit_func(x_data)
                                ss_res = np.sum((y_data - y_pred) ** 2)
                                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                
                                # Store for statistics
                                config['r_squared'] = r_squared
                                config['equation'] = format_polynomial(coeffs, degree)
                except:
                    pass
        
        if not curve_data:
            return None, "❌ No valid data in any dataset", "", "", "", ""
        
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
        
        # Layout configuration
        layout_options = {
            'height': 600,
            'margin': dict(l=50, r=50, b=50, t=50, pad=4),
            'showlegend': True
        }
        
        # Grid options
        if "Show Grid" in plot_options:
            layout_options['xaxis_showgrid'] = True
            layout_options['yaxis_showgrid'] = True
        
        # Labels
        if "Show Axes Labels" in plot_options:
            x_label = custom_x_label.strip() if custom_x_label and custom_x_label.strip() else "X values"
            y_label = custom_y_label.strip() if custom_y_label and custom_y_label.strip() else "Y values"
            layout_options['xaxis_title'] = x_label
            layout_options['yaxis_title'] = y_label
        
        # Title
        if "Show Title" in plot_options:
            title = custom_title.strip() if custom_title and custom_title.strip() else "Multi-Dataset Analysis"
            layout_options['title'] = title
        
        # Axis scaling
        layout_options['xaxis_type'] = xaxis_scale
        layout_options['yaxis_type'] = yaxis_scale
        
        # Font customization
        safe_font_family = font_family if font_family else "Arial"
        safe_title_size = max(8, min(32, title_font_size)) if title_font_size else 16
        safe_axes_size = max(8, min(24, axes_font_size)) if axes_font_size else 12
        safe_legend_size = max(8, min(20, legend_font_size)) if legend_font_size else 10
        
        layout_options['font'] = dict(
            family=safe_font_family,
            size=safe_axes_size,
            color=font_color if font_color else "#000000"
        )
        
        fig.update_layout(**layout_options)
        
        info_msg = f"✅ **Multi-Dataset Plot**: {len(curve_data)} datasets displayed successfully"
        
        return fig, info_msg, derivative_text_output, statistics_output, area_output, extrapolation_output
        
    except Exception as e:
        return None, f"🔴 **Plotting Error**: {str(e)}", "", "", "", ""

# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 Advanced Multi-Dataset Analysis & Polynomial Fitting Tool")
    gr.Markdown(
        "📁 **Upload CSV/XLSX files** or **enter data manually**. "
        "🎯 **Add multiple datasets** to the same chart for comparison. "
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
        'file_df': None,
        'x_col_name': None,
        'y_col_name': None,
        'z_col_name': None,
        'visible': True
    }])
    curve_names_state = gr.State(['Dataset 1'])
    current_curve_idx_state = gr.State(0)

    with gr.Row():
        with gr.Column(scale=1):
            # Dataset Management Section
            gr.Markdown("### 📊 Dataset Management")
            with gr.Row():
                curve_selector = gr.Dropdown(
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
                        info="Optional: Z values will be used for color mapping of data points"
                    )
                
                with gr.TabItem("📁 File Upload"):
                    file_input = gr.File(
                        label="Upload Data File", 
                        file_types=[".csv", ".xlsx", ".xls"],
                        file_count="single"
                    )
                    
                    with gr.Row():
                        x_column_dropdown = gr.Dropdown(
                            label="X Column", 
                            choices=[], 
                            value=None,
                            interactive=True
                        )
                        y_column_dropdown = gr.Dropdown(
                            label="Y Column", 
                            choices=[], 
                            value=None,
                            interactive=True
                        )
                    
                    z_column_dropdown = gr.Dropdown(
                        label="Z Column (Optional)", 
                        choices=[], 
                        value=None,
                        interactive=True,
                        info="Optional: Select column for color mapping"
                    )
                    
                    file_status_output = gr.Textbox(
                        label="File Status", 
                        interactive=False, 
                        lines=2
                    )

            file_dataframe_state = gr.State(value=None)

            gr.Markdown("### ⚙️ Dataset Settings")
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
            
            # Error bars
            show_error_bars_checkbox = gr.Checkbox(label="Show Error Bars", value=False)
            x_errors_input = gr.Textbox(label="X Uncertainties", placeholder="Optional", lines=1)
            y_errors_input = gr.Textbox(label="Y Uncertainties", placeholder="Optional", lines=1)

        with gr.Column(scale=2):
            gr.Markdown("### 📈 Multi-Dataset Visualization")
            plot_output = gr.Plot(label="Combined Plot", show_label=True)
            
            equation_output = gr.Textbox(
                label="Plot Status", 
                interactive=False, 
                lines=2
            )
            
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
                        lines=10,
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

    curve_selector.change(
        fn=switch_curve,
        inputs=[curve_configs_state, curve_names_state, curve_selector],
        outputs=[
            current_curve_idx_state, x_input, y_input, z_input, degree_slider,
            data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
            fit_type_dropdown, show_fit_checkbox, show_data_checkbox,
            x_errors_input, y_errors_input, show_error_bars_checkbox, visible_checkbox,
            curve_status
        ]
    )

    # File upload handling
    file_input.change(
        fn=handle_file_change,
        inputs=[file_input],
        outputs=[x_column_dropdown, y_column_dropdown, z_column_dropdown, file_dataframe_state, x_input, y_input, file_status_output]
    )

    # Main update function
    update_button = gr.Button("🚀 Update Multi-Dataset Plot", variant="primary", size="lg")
    
    update_inputs = [
        curve_configs_state, plot_options_checkbox, xaxis_scale_dropdown, yaxis_scale_dropdown, 
        special_points_checkbox, x_derivative_input, custom_x_label_input, custom_y_label_input, 
        custom_title_input, font_family_dropdown, title_font_size_slider, axes_font_size_slider, 
        legend_font_size_slider, font_color_picker, area_start_x_input, area_end_x_input, 
        show_area_checkbox, n_extrapolation_steps_input, extrapolation_step_size_input, show_extrapolation_checkbox
    ]
    
    update_button.click(
        fn=update_combined_plot,
        inputs=update_inputs,
        outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output]
    )

    # Auto-save current dataset configuration
    def auto_save_curve():
        return save_current_curve_config

    for component in [x_input, y_input, z_input, degree_slider, data_color_picker, fit_color_picker, 
                      show_fit_checkbox, show_data_checkbox, visible_checkbox]:
        component.change(
            fn=save_current_curve_config,
            inputs=[curve_configs_state, current_curve_idx_state, x_input, y_input, z_input, degree_slider,
                   data_color_picker, fit_color_picker, data_marker_dropdown, fit_line_dropdown,
                   fit_type_dropdown, show_fit_checkbox, show_data_checkbox, x_errors_input,
                   y_errors_input, show_error_bars_checkbox, visible_checkbox,
                   file_dataframe_state, x_column_dropdown, y_column_dropdown, z_column_dropdown],
            outputs=[curve_configs_state]
        )

    # Initial load
    demo.load(
        fn=update_combined_plot,
        inputs=update_inputs,
        outputs=[plot_output, equation_output, derivative_output_text, statistics_output, area_output, extrapolation_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()
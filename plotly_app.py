import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import tempfile
import os

# Helper function to parse input strings into numpy arrays
def parse_input_data(text_data):
    """Parses comma, space, or newline separated numbers into a numpy array."""
    if not text_data.strip():
        return np.array([])
    # Replace commas and newlines with spaces, then split
    numbers_str = re.split(r'[,\s\n]+', text_data.strip())
    try:
        # Filter out empty strings resulting from multiple delimiters
        numbers = [float(n) for n in numbers_str if n]
        return np.array(numbers)
    except ValueError:
        # Handle cases where conversion to float fails
        return None # Indicate error

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

# Main function to generate plot, fit polynomial, and prepare download
def update_plot_and_fit(x_text, y_text, degree, plot_options, save_dpi):
    """
    Parses data, fits polynomial, generates plot, formats equation,
    and creates a downloadable image file.
    """
    x_data = parse_input_data(x_text)
    y_data = parse_input_data(y_text)

    # --- Input Validation ---
    if x_data is None or y_data is None:
        return None, "Error: Invalid characters in input data. Please use numbers separated by commas, spaces, or newlines.", None
    if len(x_data) == 0 or len(y_data) == 0:
        return None, "Please enter X and Y data.", None
    if len(x_data) != len(y_data):
        return None, f"Error: X data has {len(x_data)} points, Y data has {len(y_data)} points. They must be the same length.", None
    if len(x_data) <= degree:
        return None, f"Error: Polynomial degree ({degree}) must be less than the number of data points ({len(x_data)}).", None
    if save_dpi <= 0:
        save_dpi = 72 # Default DPI if invalid
        error_msg = "Warning: Invalid DPI, using default 72."
    else:
        error_msg = "" # No error initially


    # --- Polynomial Fitting ---
    try:
        coeffs = np.polyfit(x_data, y_data, degree)
        poly_func = np.poly1d(coeffs)
        equation_str = format_polynomial(coeffs, degree)

        # Generate points for the fitted curve
        x_fit = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit = poly_func(x_fit)

    except Exception as e:
         return None, f"Error during polynomial fitting: {e}", None

    # --- Plotting ---
    try:
        fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figsize as needed

        # Plot original data
        ax.scatter(x_data, y_data, label='Original Data', color='blue', s=20) # s is marker size

        # Plot fitted curve
        ax.plot(x_fit, y_fit, label=f'Fit (degree {degree})', color='green', linewidth=2)

        # Apply plot options
        if "Show Grid" in plot_options:
            ax.grid(True, linestyle='--', alpha=0.6)
        if "Show Axes Labels" in plot_options:
            ax.set_xlabel("X values")
            ax.set_ylabel("Y values")
        if "Show Title" in plot_options:
             ax.set_title("Data and Polynomial Fit")

        ax.legend()
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # --- Prepare File for Download ---
        # Create a temporary file to save the plot
        # Use a context manager for cleaner file handling if possible,
        # but Gradio's File output needs a persistent path until downloaded.
        temp_dir = tempfile.gettempdir()
        # Ensure unique filename to avoid conflicts in multi-user scenarios
        temp_filename = next(tempfile._get_candidate_names()) + ".png"
        save_path = os.path.join(temp_dir, temp_filename)

        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Combine potential DPI warning with equation
        full_output_message = f"{equation_str}\n{error_msg}".strip()

        # Return plot object for display, the equation string, and the file path for download
        return fig, full_output_message, save_path # Gradio handles fig object directly for gr.Plot

    except Exception as e:
        plt.close(fig) # Ensure figure is closed even on error
        return None, f"Error during plotting or saving: {e}", None


# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ВСТАВКА МАССИВА / Array Input & Polynomial Fit")
    gr.Markdown("Paste your X and Y data below (comma, space, or newline separated). The app will plot the data and fit a polynomial curve.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Input Data & Settings**")
            x_input = gr.Textbox(
                label="X values",
                placeholder="e.g., 1, 2, 3, 4, 5\nor\n1 2 3 4 5",
                lines=5,
                value="0, 1, 2, 3, 4, 5, 6" # Example data
            )
            y_input = gr.Textbox(
                label="Y values",
                placeholder="e.g., 0.5, 2.1, 3.8, 8.2, 12.5",
                lines=5,
                value="-1.2, 0, 0.6, 1.2, 2.4, 5.0, 9.8" # Example data matching sketch polynomial roughly
            )
            degree_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3, # Default degree from sketch
                label="Polynomial Degree (n)"
            )
            equation_output = gr.Textbox(label="Fitted Polynomial Function", interactive=False)


        with gr.Column(scale=2):
            gr.Markdown("**Plot Output & Controls**")
            plot_output = gr.Plot(label="Data and Fitted Curve")

            with gr.Row():
                 plot_options_checkbox = gr.CheckboxGroup(
                     ["Show Grid", "Show Axes Labels", "Show Title"],
                     label="Plot Options",
                     value=["Show Grid", "Show Axes Labels", "Show Title"] # Default checked options
                 )
            with gr.Row():
                 save_dpi_input = gr.Number(value=480, label="Save Image DPI", minimum=50, step=10) # Default DPI from sketch
                 # Using gr.File for download
                 download_button = gr.File(label="Download Plot Image (.png)", file_count="single", type="filepath")


    # --- Connect Components ---
    inputs = [x_input, y_input, degree_slider, plot_options_checkbox, save_dpi_input]
    outputs = [plot_output, equation_output, download_button] # Plot, Equation text, File path

    # Trigger update on changes to any input
    for input_comp in inputs:
        input_comp.change(fn=update_plot_and_fit, inputs=inputs, outputs=outputs)

    # Initial load trigger
    demo.load(fn=update_plot_and_fit, inputs=inputs, outputs=outputs)


# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()
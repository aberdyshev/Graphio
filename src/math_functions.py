"""
Mathematical Functions and Models Module

This module contains all the mathematical functions, fitting models, and calculations.
"""

import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
import warnings
from typing import Tuple, Optional, Union, Callable


def format_polynomial(coeffs: np.ndarray, degree: int) -> str:
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


def exponential_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential function: f(x) = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def logarithmic_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logarithmic function: f(x) = a * ln(b * x) + c"""
    return a * np.log(b * x) + c


def power_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power function: f(x) = a / (x^b)"""
    return a / (x ** b)


def fit_alternative_model(x_data: np.ndarray, y_data: np.ndarray, model_type: str) -> Tuple[Optional[Callable], Optional[np.ndarray], Optional[np.ndarray], str]:
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
                return None, None, None, "Logarithmic fitting requires all X values to be positive"
            
            popt, pcov = curve_fit(logarithmic_func, x_data, y_data, 
                                 p0=[1, 1, 0], maxfev=5000)
            return logarithmic_func, popt, pcov, f"f(x) = {popt[0]:.3f} * ln({popt[1]:.3f} * x) + {popt[2]:.3f}"
            
        elif model_type == "power":
            # Check for positive x values
            if np.any(x_data <= 0):
                return None, None, None, "Power fitting requires all X values to be positive"
            
            popt, pcov = curve_fit(power_func, x_data, y_data, 
                                 p0=[1, 1], maxfev=5000)
            return power_func, popt, pcov, f"f(x) = {popt[0]:.3f} / (x^{popt[1]:.3f})"
            
        else:
            return None, None, None, f"Unknown model type: {model_type}"
            
    except Exception as e:
        return None, None, None, f"Fitting failed: {str(e)}"


def calculate_extrapolation(func: Callable, x_data: np.ndarray, n_steps: int, step_size: Optional[float] = None, model_params: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Calculate extrapolated values."""
    try:
        if len(x_data) == 0:
            return None, None, "No data available for extrapolation"
        
        x_max = np.max(x_data)
        
        # Determine step size
        if step_size is None or step_size <= 0:
            x_range = np.max(x_data) - np.min(x_data)
            step_size = x_range / len(x_data) if len(x_data) > 1 else 1.0
        
        # Generate extrapolation points
        x_extrap = np.array([x_max + (i + 1) * step_size for i in range(n_steps)])
        
        # Calculate y values
        if model_params is not None:
            y_extrap = func(x_extrap, *model_params)
        else:
            y_extrap = func(x_extrap)
        
        # Check for numerical issues
        if np.any(np.isnan(y_extrap)) or np.any(np.isinf(y_extrap)):
            return None, None, "Extrapolation resulted in invalid values (NaN or Inf)"
        
        return x_extrap, y_extrap, None
        
    except Exception as e:
        return None, None, f"Extrapolation error: {str(e)}"


def calculate_area_under_curve(poly_func: Callable, x_start: float, x_end: float, method: str = 'quad') -> Tuple[Optional[float], Optional[str]]:
    """Calculate area under polynomial curve between x_start and x_end."""
    try:
        if method == 'quad':
            area, error = integrate.quad(poly_func, x_start, x_end)
            
            # Check if integration was successful
            if abs(error) > abs(area) * 0.01:  # Error > 1% of area
                return area, f"⚠️ Large integration error: {error:.2e}"
            
            return area, None
        else:
            return None, f"Unsupported integration method: {method}"
            
    except Exception as e:
        return None, f"Integration error: {str(e)}"

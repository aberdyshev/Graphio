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
    """Logarithmic function with sign preservation: f(x) = sign(x) * a * ln(b * |x|) + c"""
    return np.sign(x) * a * np.log(b * np.abs(x)) + c


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
        # Проверка на ненулевые значения
            if np.all(x != 0 for x in x_data):
                try:
                    # Начальные приближения с учетом знака
                    p0 = [1.0, 1.0, 0.0]
                    popt, pcov = curve_fit(logarithmic_func, x_data, y_data, 
                                         p0=p0, maxfev=5000)
                    return logarithmic_func, popt, pcov, f"f(x) = sign(x)*{popt[0]:.3f} * ln({popt[1]:.3f}*|x|) + {popt[2]:.3f}"
                except RuntimeError:
                    return None, None, None, "Ошибка аппроксимации (возможно, данные не подходят)"
            else:
                return None, None, None, "Логарифмическая аппроксимация требует X ≠ 0"
            
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


def calculate_extrapolation(fit_func, x_data, n_steps, step_size=None, model_params=None):
    """Calculate extrapolation points starting immediately after the last data point"""
    if len(x_data) == 0:
        return None, None, "No data available for extrapolation"
    
    try:
        last_x = x_data[-1]
        
        # Calculate automatic step size if not provided
        if step_size is None or step_size <= 0:
            if len(x_data) > 1:
                step_size = np.mean(np.diff(x_data))  # Средний шаг исходных данных
            else:
                step_size = 1.0  # Значение по умолчанию для одного элемента
        
        # Генерируем точки экстраполяции НАЧИНАЯ С последней точки
        x_extrap = np.array([last_x + i*step_size for i in range(1, n_steps+1)])
        
        if model_params is not None:
            y_extrap = fit_func(x_extrap, *model_params)
        else:
            y_extrap = fit_func(x_extrap)
            
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

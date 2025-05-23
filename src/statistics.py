"""
Statistics and Analysis Module

This module handles statistical calculations and analysis of data.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_statistics(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Optional[Dict], Optional[str]]:
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
            stats['correlation'] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        else:
            stats['correlation'] = 0.0
        
        # Number of data points
        stats['n_points'] = len(x_data)
        
        # Correlation strength interpretation
        corr = abs(stats['correlation'])
        if corr >= 0.9:
            stats['corr_strength'] = "Very Strong"
        elif corr >= 0.7:
            stats['corr_strength'] = "Strong"
        elif corr >= 0.5:
            stats['corr_strength'] = "Moderate"
        elif corr >= 0.3:
            stats['corr_strength'] = "Weak"
        elif corr > 0:
            stats['corr_strength'] = "Very Weak"
        else:
            stats['corr_strength'] = "N/A"
        
        # Correlation direction
        if stats['correlation'] > 0.1:
            stats['corr_direction'] = "Positive"
        elif stats['correlation'] < -0.1:
            stats['corr_direction'] = "Negative"
        else:
            stats['corr_direction'] = "None/Weak"
        
        return stats, None
        
    except Exception as e:
        return None, f"Statistics calculation error: {str(e)}"


def format_statistics_output(stats: Dict) -> str:
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
        if stats['corr_strength'] in ["Very Strong", "Strong"]:
            output += f"\n• ✅ **Interpretation**: {stats['corr_strength'].lower()} {stats['corr_direction'].lower()} relationship"
        elif stats['corr_strength'] == "Moderate":
            output += f"\n• 👌 **Interpretation**: Moderate {stats['corr_direction'].lower()} relationship"
        else:
            output += f"\n• ⚠️ **Interpretation**: {stats['corr_strength']} {stats['corr_direction'].lower()} relationship"

    return output


def format_area_output(area: float, x_start: float, x_end: float, poly_func=None) -> str:
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


def format_extrapolation_output(x_extrap: np.ndarray, y_extrap: np.ndarray, n_steps: int) -> str:
    """Format extrapolation results into a beautiful display string."""
    if x_extrap is None or y_extrap is None:
        return "❌ Unable to calculate extrapolation"
    
    output = f"""🔮 **Extrapolation Results** ({n_steps} steps ahead)

**📈 Predicted Values:**
"""
    
    for i, (x, y) in enumerate(zip(x_extrap, y_extrap)):
        output += f"• Step {i+1}: X = {x:.4f}, Y = {y:.4f}\n"
    
    # Add trend analysis
    if len(y_extrap) > 1:
        trend = "increasing" if y_extrap[-1] > y_extrap[0] else "decreasing"
        rate = abs(y_extrap[-1] - y_extrap[0]) / (len(y_extrap) - 1)
        output += f"\n**📊 Trend Analysis:**\n• Direction: {trend.title()}\n• Average rate of change: {rate:.4f} per step"
    
    return output

"""Temporal filtering and smoothing utilities."""

import numpy as np
from typing import Optional


class OneEuroFilter:
    """One Euro Filter for temporal smoothing.
    
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """Initialize One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (decrease to reduce jitter)
            beta: Speed coefficient (increase to reduce lag)
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate smoothing factor."""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(self, a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing."""
        return a * x + (1 - a) * x_prev
    
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Apply filter to new measurement.
        
        Args:
            x: New measurement (can be scalar or array)
            t: Timestamp in seconds
        
        Returns:
            Filtered value
        """
        x = np.asarray(x, dtype=np.float64)
        
        # First call - initialize
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        
        # Calculate time delta
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-6  # Prevent division by zero
        
        # Calculate derivative
        dx = (x - self.x_prev) / t_e
        
        # Smooth derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        # Calculate adaptive cutoff (use mean for multi-dimensional arrays)
        dx_magnitude = np.mean(np.abs(dx_hat)) if dx_hat.ndim > 0 else np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * dx_magnitude
        
        # Smooth value
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t
        
        return x_hat


class MultiChannelOneEuroFilter:
    """One Euro Filter for multi-channel data (e.g., 3D positions)."""
    
    def __init__(self, n_channels: int, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """Initialize multi-channel filter.
        
        Args:
            n_channels: Number of channels
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative
        """
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_channels)]
    
    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()
    
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Apply filter to multi-channel measurement.
        
        Args:
            x: New measurement, shape (n_channels,)
            t: Timestamp in seconds
        
        Returns:
            Filtered values, shape (n_channels,)
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for i, f in enumerate(self.filters):
            result[i] = f(x[i], t)
        return result


def savitzky_golay_filter(data: np.ndarray, window_length: int = 5, polyorder: int = 2, axis: int = 0) -> np.ndarray:
    """Apply Savitzky-Golay filter for smoothing.
    
    Args:
        data: Input data
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial
        axis: Axis along which to apply the filter
    
    Returns:
        Filtered data
    """
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_length, polyorder, axis=axis)
    except ImportError:
        # Fallback: simple moving average
        return moving_average_filter(data, window_length, axis=axis)


def moving_average_filter(data: np.ndarray, window_length: int = 5, axis: int = 0) -> np.ndarray:
    """Apply simple moving average filter.
    
    Args:
        data: Input data
        window_length: Length of the filter window
        axis: Axis along which to apply the filter
    
    Returns:
        Filtered data
    """
    try:
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(data, size=window_length, axis=axis, mode='nearest')
    except ImportError:
        # Fallback: simple numpy-based moving average
        kernel = np.ones(window_length) / window_length
        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            # Apply along specified axis
            result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis, data)
            return result

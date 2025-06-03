import numpy as np

def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute moving average ignoring NaNs.
    Uses 'same' convolution so output has the same length as input.

    Args:
        y: The input array.
        window: The window size for the moving average.
            
    Returns:
        The moving average of the input array, same length as y.
    """
    if window < 1 or window > len(y):
        raise ValueError("Window size must be between 1 and length of y.")
    y_filled = np.nan_to_num(y, nan=np.nanmean(y))
    kernel = np.ones(window) / window
    y_smoothed = np.convolve(y_filled, kernel, mode="same")
    return y_smoothed

def normalize(y: np.ndarray) -> np.ndarray:
    """
    Scale values to [0, 1], ignoring NaNs.
    """
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    if ymax > ymin:
        return (y - ymin) / (ymax - ymin) + 1e-8
    else:
        return y

def fill_nan_with_interp(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 1D array using linear interpolation.
    """
    x = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan_to_num(arr, nan=0.0)
    return np.interp(x, x[mask], arr[mask])
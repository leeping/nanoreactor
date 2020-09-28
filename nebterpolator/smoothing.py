"""
Smoothing a 1d signal
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np
from scipy.optimize import leastsq
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

##############################################################################
# Globals
##############################################################################

__all__ = ['polynomial_smooth', 'window_smooth', 'buttersworth_smooth']

##############################################################################
# Functions
##############################################################################


def polynomial_smooth(y, x=None, order=2, end_weight=1):
    """Smooth a dataset by fitting it to a polynomial

    Parameters
    ----------
    y : np.ndarray
        The signal
    x : np.ndarray, optional
        The x coordinate of each point. If left unsupplied, we'll
        take the x range to be just the ints 0 through len(y)-1
    order : int
        The order of the polynomial

    Returns
    -------
    smoothed : np.ndarray
        The value of the fitted polynomial at each point x
    """
    if x is None:
        x = np.arange(len(y))

    weights = np.r_[end_weight, np.ones(len(x)-2), end_weight]

    def func(p):
        return (np.polyval(p, x) - y) * weights

    # need 1 more for the constant, so that order 2 is quadratic
    # (even though it's 3 params)
    #popt, pcov = curve_fit(func, x, y, p0=np.ones(order+1), sigma=1.0/weights)
    popt, covp, info, msg, ier = leastsq(func, x0=np.zeros(order+1),
                                         full_output=True)
    return np.polyval(popt, x)


def window_smooth(signal, window_len=11, window='flat'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    This code is copied from the scipy cookbook, with sytlistic improvements.
    http://www.scipy.org/Cookbook/SignalSmooth

    Parameters
    ----------
    signal : np.ndarray, ndim=1
        The input signal
    window_len: int
        The dimension of the smoothing window; should be an odd integer
    window: {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        Which type of window to use? Flat will produce a moving average
        smoothin

    Returns
    -------
    output : np.ndarray, ndim=1
        The smoothed signal
    """

    if signal.ndim != 1:
        raise TypeError('I only smooth 1d arrays')
    if signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len % 2 != 1:
        raise ValueError('window_len must be an odd integer')
    if window_len < 3:
        return signal

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'trivial']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman', 'trivial'")

    # this does a mirroring padding
    padded = np.r_[2*signal[0] - signal[window_len-1: 0: -1],
                   signal,
                   2*signal[-1] - signal[-2: -window_len-1: -1]]

    if window == 'trivial':
        output = np.linspace(signal[0], signal[-1], len(signal))
        return output
    else:
        if window == 'flat':
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, window)(window_len)
        output = np.convolve(w / w.sum(), padded, mode='valid')
        return output[(window_len//2):-(window_len//2)] #Not sure whether this needs to be integer or not.


def buttersworth_smooth(signal, width=11, order=3):
    """Smooth the data using zero-delay buttersworth filter

    This code is copied from the scipy cookbook, with sytlistic improvements.
    http://www.scipy.org/Cookbook/FiltFilt

    Parameters
    ----------
    signal : np.ndarray, ndim=1
        The input signal
    width : float
        This acts very similar to the window_len in the window smoother. In
        the implementation, the frequency of the low-pass filter is taken to
        be two over this width, so it's like "half the period" of the sinusiod
        where the filter starts to kick in.
    order : int, optional
        The order of the filter. A small odd number is recommended. Higher
        order filters cutoff more quickly, but have worse numerical
        properties.

    Returns
    -------
    output : np.ndarray, ndim=1
        The smoothed signal
    """
    if width < 2.0:
        return signal

    # first pad the signal on the ends
    pad = int(np.ceil((width + 1)/2)*2 - 1)  # nearest odd integer
    padded = np.r_[signal[pad - 1: 0: -1], signal, signal[-1: -pad: -1]]
    #padded = np.r_[[signal[0]]*pad, signal, [signal[-1]]*pad]

    b, a = butter(order, 2.0 / width)
    # Apply the filter to the width.  Use lfilter_zi to choose the
    # initial condition of the filter.
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, padded, zi=zi*padded[0])
    # Apply the filter again, to have a result filtered at an order
    # the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi*z[0])
    # Use filtfilt to apply the filter.
    output = filtfilt(b, a, padded)

    return output[(pad-1): -(pad-1)]


def angular_smooth(signal, smoothing_func=buttersworth_smooth, **kwargs):
    """Smooth an signal which represents an angle by filtering its
    sine and cosine components separately.

    Parameters
    ----------
    signal : np.ndarray, ndim=1
        The input signal
    smoothing_func : callable
        A function that takes the signal as its first argument and smoothes
        it.

    All other parameters (**kwargs) will be passed through to smoothing_func.

    Returns
    -------
    smoothed_signal : bp.ndarray, ndim=1
        The smoothed version of the function.
    """
    sin = smoothing_func(np.sin(signal), **kwargs)
    cos = smoothing_func(np.cos(signal), **kwargs)
    return np.arctan2(sin, cos)


def main():
    "test code"
    import matplotlib.pyplot as pp
    N = 1000
    sigma = 0.25
    x = np.cumsum(sigma * np.random.randn(N))
    y = np.cumsum(sigma * np.random.randn(N))
    signal = np.arctan2(x, y)
    pp.plot(signal)

    pp.plot(np.arctan2(filtfit_smooth(np.sin(signal), width=21),
                       filtfit_smooth(np.cos(signal), width=21)))

    pp.show()


if __name__ == '__main__':
    main()

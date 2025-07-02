'''Useful MuPT functions that assist with calculations done in other scripts.
   Basic geometry or functions that don't fit into a specific category belong here!'''

__author__ = 'Joe Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

#################################################################################################
import numpy as np

#### Helper function from cmeutils.plotting.py to get histogram data
def get_histogram(data, normalize=False, bins="auto", x_range=None):
    """Bins a 1-D array of data into a histogram using
    the numpy.histogram method.

    Parameters
    ----------
    data : 1-D numpy.array, required
        Array of data used to generate the histogram
    normalize : boolean, default=False
        If set to true, normalizes the histogram bin heights
        by the sum of data so that the distribution adds
        up to 1
    bins : float, int, or str, default="auto"
        Method used by numpy to determine bin borders.
        Check the numpy.histogram docs for more details.
    x_range : (float, float), default = None
        The lower and upper range of the histogram bins.
        If set to None, then the min and max values of data are used.

    Returns
    -------
    bin_cetners : 1-D numpy.array
        Array of the bin center values
    bin_heights : 1-D numpy.array
        Array of the bin height values

    """
    bin_heights, bin_borders = np.histogram(
        a=data, bins=bins, range=x_range, density=normalize
    )
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    return bin_centers, bin_heights
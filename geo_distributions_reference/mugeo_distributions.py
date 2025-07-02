'''MuPT functions for calculating geometric property distributions for 
   CG simulations run in the MuPT ecosystem.'''

__author__ = 'Joe Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

#################################################################################################

from mutils.mutils import get_histogram
import MDAnalysis as mda
import numpy as np
import warnings

# Bond Distribution
# The following function calculates the distribution of bond lengths in a CG simulation
# for a given set of atoms. User must input which two particle types they are 
# interested in computing the bond data for.

def mupt_bond_distribution(
    gsd_file: str,
    A_name: str,
    B_name: str,
    start: int=0,
    stop: int=-1,
    histogram: bool=False,
    l_min: float=0.0,
    l_max: float=4.0,
    normalize: bool=True,
    bins: int=100,
):
    """Returns the bond length distribution for a given bond pair

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name : str
        Name(s) of particles that form the bond pair
        (found in gsd.hoomd.Frame.particles.types)
    start : int
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end. (default 0)
    stop : int
        Final frame index for accumulating bond lengths. (default -1)
    histogram : bool, default=False
        If set to True, places the resulting bonds into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted bonds.
    l_min : float, default = 0.0
        Sets the minimum bond length to be included in the distribution
    l_max : float, default = 5.0
        Sets the maximum bond length value to be included in the distribution
    normalize : bool, default=False
        If set to True, normalizes the angle distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
    bins : float, int, or str,  default="auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual bond angles in degrees
        If histogram is True, returns a 2D array of bin centers and bin heights.

    """
    # Load the GSD file using MDAnalysis
    u = mda.Universe(gsd_file)

    # generate the selection string so we only iterate over the atoms
    # that can possibly participate in the desired bond
    # I.E, if you have particles of type A, B, and C,
    # You don't want to consider bonds that involve B and C atoms

    selection_string = f"name {A_name} or name {B_name}"

    all_bonds = []

    # Select atoms based on the selection string
    for ts in u.trajectory[start:stop]:
        relevent_atoms = u.select_atoms(selection_string)

        # Iterate over all of the bonds in the selection from the universe
        for bond in relevent_atoms.bonds:
            bonded_atoms = list(bond.atoms.names)

            # Check if the bond contains both A_name and B_name
            if A_name in bonded_atoms and B_name in bonded_atoms:
                # if verbose:
                #     #print(f"Bond: {bond}, Atoms: {bond.atoms}, Length: {bond.length()}")
                # else:
                #     #print(f"Bond Length: {bond.length()} between atoms {bond.atoms.ix}")
                all_bonds.append(bond.length())

    if histogram:
        if min(all_bonds) < l_min or max(all_bonds) > l_max:
            warnings.warn(
                "There are bond lengths that fall outside of "
                "your set l_min and l_max range. You may want to adjust "
                "this range to include all bond lengths."
            )
        bin_centers, bin_heights = get_histogram(
            data=np.array(all_bonds),
            normalize=normalize,
            bins=bins,
            x_range=(l_min, l_max),
        )
        return np.stack((bin_centers, bin_heights)).T
    else:
        return np.array(all_bonds)

















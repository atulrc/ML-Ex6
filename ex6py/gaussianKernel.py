# Scientific and vector computation for python
import numpy as np

def gaussKer(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.

    Parameters
    ----------
    x1 :  numpy ndarray
        A vector of size (n, ), representing the first datapoint.

    x2 : numpy ndarray
        A vector of size (n, ), representing the second datapoint.

    sigma : float
        The bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    sim : float
        The computed RBF between the two provided data points.

    Instructions
    ------------
    Fill in this function to return the similarity between `x1` and `x2`
    computed using a Gaussian kernel with bandwidth `sigma`.
    """
    sim = 0
    # ====================== YOUR CODE HERE ======================

    sim = np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))

    # =============================================================
    return sim

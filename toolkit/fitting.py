# Licensed under the MIT License - see LICENSE.rst
"""
Methods for fitting transit light curves, spot occultations, or both, using
`scipy` minimizers and `emcee`.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import emcee
from scipy import optimize, signal
import matplotlib.pyplot as plt
import batman
from copy import deepcopy
from emcee.utils import MPIPool
import sys



def generate_lc(times, transit_params):
    """
    Make a transit light curve.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times in JD
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    model_flux : `numpy.ndarray`
        Fluxes from model transit light curve
    """
    exp_time = 1./60/24  # 1 minute cadence -> [days]

    m = batman.TransitModel(transit_params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params)
    return model_flux



def lnprior(theta, y, lower_t_bound, upper_t_bound, transit_params,
            skip_priors):
    """
    Log prior for `emcee` runs.

    Parameters
    ----------
    theta : list
        Fitting parameters
    y : `numpy.ndarray`
        Fluxes
    lower_t_bound : float
        Earliest in-transit time [JD]
    upper_t_bound : float
        Latest in-transit time [JD]
    skip_priors : bool
        Should the priors be skipped?

    Returns
    -------
    lnpr : float
        Log-prior for trial parameters `theta`
    """
    spot_params = theta

    amplitudes = spot_params[::3]
    t0s = spot_params[1::3]
    sigmas = spot_params[2::3]
    depth = transit_params.rp**2

    min_sigma = 1.0/60/24
    max_sigma = transit_params.duration  # 6.0e-3  # upper_t_bound - lower_t_bound
    t0_ok = ((lower_t_bound < t0s) & (t0s < upper_t_bound)).all()
    sigma_ok = ((min_sigma < sigmas) & (sigmas < max_sigma)).all()
    if not skip_priors:
        amplitude_ok = ((0 <= amplitudes) & (amplitudes < depth)).all()
    else:
        amplitude_ok = (amplitudes >= 0).all()

    if amplitude_ok and t0_ok and sigma_ok:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr, transit_params, skip_priors=False):
    """
    Log-likelihood of data given model.

    Parameters
    ----------
    theta : list
        Trial parameters
    x : `numpy.ndarray`
        Times in JD
    y : `numpy.ndarray`
        Fluxes
    yerr : `numpy.ndarray`
        Uncertainties on fluxes
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    lnp : float
        Log-likelihood of data given model, i.e. ln( P(x | theta) )
    """
    model = spotted_transit_model(theta, x, transit_params, skip_priors)
    return -0.5*np.sum((y-model)**2/yerr**2)


def lnprob(theta, x, y, yerr, lower_t_bound, upper_t_bound, transit_params,
           skip_priors):
    """
    Log probability.

    Parameters
    ----------
    theta : list
        Trial parameters
    x : `numpy.ndarray`
        Times in JD
    y : `numpy.ndarray`
        Fluxes
    yerr : `numpy.ndarray`
        Uncertainties on fluxes
    lower_t_bound : float
        Earliest in-transit time [JD]
    upper_t_bound : float
        Latest in-transit time [JD]
    transit_params : `~batman.TransitParams`
        Transit light curve parameters
    Returns
    -------

    """
    lp = lnprior(theta, y, lower_t_bound, upper_t_bound, transit_params,
                 skip_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, transit_params, skip_priors)


    return sampler

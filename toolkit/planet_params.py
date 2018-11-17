from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import batman

__all__ = ['hat7_params', 'tres2_params', 'kepler91_params', 'kepler13_params',
           'kepler412_params', 'kepler39_params', 'kepler435_params']


def hat7_params():
    """
    Assumed transit parameters for HAT-P-7 b from exoplanets.org [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/HAT-P-7_b
    """
    eccentricity = 0.0
    omega = 90

    params = batman.TransitParams()
    params.t0 = 2454954.35853331    # time of inferior conjunction
    params.per = 2.2047354         # orbital period
    params.rp = 0.00601399**0.5          # planet radius (in units of stellar radii)
    dur = 0.16430159                 # transit duration
    params.inc = 83.143             # orbital inclination (in degrees)
    params.b = 0.49723805
    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 4.1545                # semi-major axis (in units of stellar radii)
    params.u = [0.3492, 0.1733]      # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    params.inc_stellar = 83.143
    params.lam = 0
    params.per_rot = params.per
    return params


def kepler412_params():
    """
    Assumed transit parameters for HAT-P-7 b from exoplanets.org [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/HAT-P-7_b
    """
    eccentricity = 0.0
    omega = 90

    params = batman.TransitParams()
    params.t0 = 2400000 + 54966.021033    # time of inferior conjunction
    params.per = 1.720861232        # orbital period
    params.rp = 0.1058          # planet radius (in units of stellar radii)
    dur = 2.077/24                 # transit duration
    params.inc = 80.89            # orbital inclination (in degrees)
    params.b = 0.781
    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 4.947                # semi-major axis (in units of stellar radii)
    params.u = [0.18, 0.51]      # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    params.inc_stellar = 90
    params.lam = 0
    params.per_rot = params.per
    return params


def tres2_params():
    """
    Assumed transit parameters for TrES-2 b from exoplanets.org [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/TrES-2_b
    """
    eccentricity = 0.0
    omega = 90

    params = batman.TransitParams()
    params.t0 = 122.763360 + 2454833     # time of inferior conjunction
    params.per = 2.47061317          # orbital period
    params.rp = 0.12539           # planet radius (in units of stellar radii)
    dur = 0.0760                  # transit duration
    params.inc = 83.872          # orbital inclination (in degrees)

    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 7.903               # semi-major axis (in units of stellar radii)
    params.u = [0.330, 0.285]     # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    return params


def kepler91_params():
    """
    Assumed transit parameters for TrES-2 b from exoplanets.org [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/TrES-2_b
    """
    eccentricity = 0.0
    omega = 90

    params = batman.TransitParams()
    params.t0 = 136.3958 + 2454833     # time of inferior conjunction
    params.per = 6.24658          # orbital period
    params.rp = 0.02181         # planet radius (in units of stellar radii)
    dur = 0.624658                  # transit duration
    params.inc = 69.68          # orbital inclination (in degrees)
    params.b = 0.8667
    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 2.496               # semi-major axis (in units of stellar radii)
    params.u = [0.73, -0.002]     # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    return params


def kepler13_params():
    """
    Assumed transit parameters for TrES-2 b from exoplanets.org [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/TrES-2_b
    """
    eccentricity = 0.0
    omega = 90

    params = batman.TransitParams()
    params.t0 = 120.56596 + 2454833     # time of inferior conjunction
    params.per = 1.763588          # orbital period
    params.rp = 0.087373         # planet radius (in units of stellar radii)
    dur = 0.16                  # transit duration
    params.inc = 86.769          # orbital inclination (in degrees)

    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 4.5008              # semi-major axis (in units of stellar radii)
    params.u = [0.3183, 0.2024]     # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    return params


def kepler39_params():
    """
    Assumed transit parameters for Kepler-39 b from Bonomo 2015

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/TrES-2_b
    """
    eccentricity = 0.112
    omega = 98.9

    params = batman.TransitParams()
    params.t0 = 1042.60708+2454900     # time of inferior conjunction
    params.per = 21.087210        # orbital period
    params.rp = 0.0896         # planet radius (in units of stellar radii)
    dur = 0.2483               # transit duration
    params.inc = 89.07          # orbital inclination (in degrees)

    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 27.74              # semi-major axis (in units of stellar radii)
    params.u = [0.31, 0.17]     # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    return params


def kepler435_params():
    """
    Assumed transit parameters for Kepler-39 b from Bonomo 2015

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters
    .. [1]  http://exoplanets.org/detail/TrES-2_b
    """
    eccentricity = 0.114
    omega = 104

    params = batman.TransitParams()
    params.t0 = 2455010.64241     # time of inferior conjunction
    params.per = 8.6001536        # orbital period
    params.rp = 0.06384         # planet radius (in units of stellar radii)
    dur = 9.003/24              # transit duration
    params.inc = 85.51          # orbital inclination (in degrees)
    params.b = 0.448
    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 6.35              # semi-major axis (in units of stellar radii)
    params.u = [0.374, 0.180]     # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    return params


# from astropy.utils.data import download_file
# from astropy.io import ascii
# class ExoplanetTable(object):
#     def __init__(self, cache=True):
#         exoplanets_url = 'http://exoplanets.org/csv-files/exoplanets.csv'
#         table_path = download_file(exoplanets_url, cache=cache)
#         table = ascii.read(table_path)
#         self.table = table
#
# EXOPLANET_TABLE = ExoplanetTable()
#
#
# class PlanetParams(object):
#     def __init__(self, exoplanet_name, cache=True):
#         table = EXOPLANET_TABLE.table
#
#         if len(np.argwhere(table['NAME'].data == exoplanet_name)) == 0:
#             raise ValueError("No planet found with name {0}"
#                              .format(exoplanet_name))
#
#         row_index = np.argwhere(table['NAME'].data == exoplanet_name)[0, 0]
#
#         for col in table.keys():
#             setattr(self, col.lower(), table[col][row_index])
#
#         self.duration = self.t14

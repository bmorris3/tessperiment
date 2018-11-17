"""
Manage the systems' orbital, planet, stellar parameters.
"""
import batman
import numpy as np


def aRs_i(transit_params):
    """
    Convert from duration and impact param to a/Rs and inclination

    Parameters
    ----------
    transit_params : `batman.TransitParams`
        Transit parameters
    Returns
    -------
    aRs : float
        Semi-major axis in units of stellar radii
    i : float
        Orbital inclination in degrees
    """
    eccentricity = transit_params.ecc
    omega = transit_params.w
    b = transit_params.b
    T14 = transit_params.duration
    P = transit_params.per
    RpRs = transit_params.rp

    # Eccentricity term for b -> a/rs conversion
    beta = (1 - eccentricity**2)/(1 + eccentricity*np.sin(np.radians(omega)))

    # Eccentricity term for duration equation:
    c = (np.sqrt(1 - eccentricity**2) /
         (1 + eccentricity*np.sin(np.radians(omega))))

    i = np.arctan(beta * np.sqrt((1 + RpRs)**2 - b**2) /
                  (b * np.sin(T14*np.pi / (P*c))))
    aRs = b/(np.cos(i) * beta)
    return aRs, np.degrees(i)


def transit_duration(transit_params):
    """
    Calculate transit duration from batman transit parameters object.

    Parameters
    ----------
    transit_params : `batman.TransitParams`
    """
    # Eccentricity term for duration equation:
    c = (np.sqrt(1 - transit_params.ecc**2) /
         (1 + transit_params.ecc*np.sin(np.radians(transit_params.w))))

    return (transit_params.per/np.pi *
            np.arcsin(np.sqrt((1 + transit_params.rp)**2 - transit_params.b**2)/
                      (np.sin(np.radians(transit_params.inc)) *
                       transit_params.a))) * c



# Licensed under the MIT License - see LICENSE.rst
"""
Methods for taking the raw light curves from MAST and producing cleaned light
curves.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import batman


def generate_lc_depth(times, depth, transit_params):
    """
    Generate a model transit light curve.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in JD
    depth : float
        Set depth independently from the setting in `transit_params`
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------

    """
    exp_time = (1*u.min).to(u.day).value

    transit_params.rp = np.sqrt(depth)

    m = batman.TransitModel(transit_params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params)
    return model_flux


class LightCurve(object):
    """
    Container object for light curves.
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None,
                 name=None):
        """
        Parameters
        ----------
        times : `~numpy.ndarray`
            Times in JD
        fluxes : `~numpy.ndarray`
            Fluxes (normalized or not)
        errors : `~numpy.ndarray`
            Uncertainties on the fluxes
        quarters : `~numpy.ndarray` (optional)
            Kepler Quarter for each flux
        name : str
            Name this light curve (optional)
        """
        # if len(times) < 1:
        #    raise ValueError("Input `times` have no length.")

        if isinstance(times[0], Time) and isinstance(times, np.ndarray):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')

        self.times = times
        self.fluxes = fluxes
        if self.times is not None and errors is None:
            errors = np.zeros_like(self.fluxes) - 1
        self.errors = errors
        if self.times is not None and quarters is None:
            quarters = np.zeros_like(self.fluxes) - 1
        self.quarters = quarters
        self.name = name

    def phases(self, params):
        phase = ((self.times.jd - params.t0) % params.per)/params.per
        phase[phase > 0.5] -= 1.0
        return phase

    def plot(self, transit_params=None, ax=None, quarter=None, show=True,
             phase=False, plot_kwargs={'color':'b', 'marker':'o', 'lw':0}):
        """
        Plot light curve.

        Parameters
        ----------
        transit_params : `~batman.TransitParams` (optional)
            Transit light curve parameters. Required if `phase` is `True`.
        ax : `~matplotlib.axes.Axes` (optional)
            Axis to make plot on top of
        quarter : float (optional)
            Plot only this Kepler quarter
        show : bool
            If `True`, call `matplotlib.pyplot.show` after plot is made
        phase : bool
            If `True`, map times in JD to orbital phases, which requires
            that `transit_params` be input also.
        plot_kwargs : dict
            Keyword arguments to pass to `~matplotlib` calls.
        """
        if quarter is not None:
            if hasattr(quarter, '__len__'):
                mask = np.zeros_like(self.fluxes).astype(bool)
                for q in quarter:
                    mask |= self.quarters == q
            else:
                mask = self.quarters == quarter
        else:
            mask = np.ones_like(self.fluxes).astype(bool)

        if ax is None:
            ax = plt.gca()

        if phase:
            x = (self.times.jd - transit_params.t0)/transit_params.per % 1
            x[x > 0.5] -= 1
        else:
            x = self.times.jd

        ax.plot(x[mask], self.fluxes[mask],
                **plot_kwargs)
        ax.set(xlabel='Time' if not phase else 'Phase',
               ylabel='Flux', title=self.name)

        if show:
            plt.show()

    def save_to(self, path, overwrite=False, for_stsp=False):
        """
        Save times, fluxes, errors to new directory ``dirname`` in ``path``
        """
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if not for_stsp:
            if os.path.exists(output_path) and overwrite:
                shutil.rmtree(output_path)

            if not os.path.exists(output_path):
                os.mkdir(output_path)
                for attr in ['times_jd', 'fluxes', 'errors', 'quarters']:
                    np.savetxt(os.path.join(path, dirname,
                                            '{0}.txt'.format(attr)),
                               getattr(self, attr))

        else:
            if not os.path.exists(output_path) or overwrite:
                attrs = ['times_jd', 'fluxes', 'errors']
                output_array = np.zeros((len(self.fluxes), len(attrs)),
                                        dtype=float)
                for i, attr in enumerate(attrs):
                    output_array[:, i] = getattr(self, attr)
                np.savetxt(os.path.join(path, dirname+'.txt'), output_array)

    @classmethod
    def from_raw_fits(cls, fits_paths, name=None):
        """
        Load FITS files downloaded from MAST into the `LightCurve` object.

        Parameters
        ----------
        fits_paths : list
            List of paths to FITS files to read in
        name : str (optional)
            Name of light curve

        Returns
        -------
        lc : `LightCurve`
            The light curve for the data in the fits files.
        """
        fluxes = []
        errors = []
        times = []
        quarter = []

        # Manual on times: http://archive.stsci.edu/kepler/manuals/archive_manual.htm

        for path in fits_paths:
            data = fits.getdata(path)
            header = fits.getheader(path)
            timslice = fits.open(path)[1].header['TIMSLICE']
            time_slice_correction = (0.25 + 0.62*(5.0 - timslice))/86400
            times.append(data['TIME'] + 2454833.0)# - data['TIMECORR'] + time_slice_correction)
            errors.append(data['SAP_FLUX_ERR'])
            fluxes.append(data['SAP_FLUX'])
            quarter.append(len(data['TIME'])*[header['QUARTER']])

        times, fluxes, errors, quarter = [np.concatenate(i)
                                          for i in [times, fluxes,
                                                    errors, quarter]]

        mask_nans = np.zeros_like(fluxes).astype(bool)
        for attr in [times, fluxes, errors]:
            mask_nans |= np.isnan(attr)

        times, fluxes, errors, quarter = [attr[~mask_nans]
                                           for attr in [times, fluxes, errors, quarter]]

        return LightCurve(times, fluxes, errors, quarters=quarter, name=name)

    @classmethod
    def from_dir(cls, path, for_stsp=False):
        """Load light curve from numpy save files in ``dir``"""
        if not for_stsp:
            times, fluxes, errors, quarters = [np.loadtxt(os.path.join(path, '{0}.txt'.format(attr)))
                                               for attr in ['times_jd', 'fluxes', 'errors', 'quarters']]
        else:
            quarters = None
            times, fluxes, errors = np.loadtxt(path, unpack=True)

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path

        if name.endswith('.txt'):
            name = name[:-4]

        return cls(times, fluxes, errors, quarters=quarters, name=name)

    def normalize_each_quarter(self, rename=None, polynomial_order=2,
                               plots=False):
        """
        Use polynomial fit to each quarter to normalize the data.

        Parameters
        ----------
        rename : str (optional)
            New name of the light curve after normalization
        polynomial_order : int (optional)
            Order of polynomial to fit to the out-of-transit fluxes. Default
            is 2.
        plots : bool (optional)
            Show diagnostic plots after normalization.
        """
        quarter_inds = list(set(self.quarters))
        quarter_masks = [quarter == self.quarters for quarter in quarter_inds]

        for quarter_mask in quarter_masks:

            polynomial = np.polyfit(self.times[quarter_mask].jd,
                                    self.fluxes[quarter_mask], polynomial_order)
            scaling_term = np.polyval(polynomial, self.times[quarter_mask].jd)
            self.fluxes[quarter_mask] /= scaling_term
            self.errors[quarter_mask] /= scaling_term

            if plots:
                plt.plot(self.times[quarter_mask], self.fluxes[quarter_mask])
                plt.show()

        if rename is not None:
            self.name = rename

    def delete_outliers(self):

        d = np.diff(self.fluxes)
        spikey = np.abs(d - np.median(d)) > 2.5*np.std(d)
        neighboring_spikes = spikey[1:] & spikey[:-1]
        opposite_signs = np.sign(d[1:]) != np.sign(d[:-1])
        outliers = np.argwhere(neighboring_spikes & opposite_signs) + 1
        #print('number bad fluxes: {0}'.format(len(outliers)))

        self.times = Time(np.delete(self.times.jd, outliers), format='jd')
        self.fluxes = np.delete(self.fluxes, outliers)
        self.errors = np.delete(self.errors, outliers)
        self.quarters = np.delete(self.quarters, outliers)

    def mask_out_of_transit(self, params, oot_duration_fraction=0.25,
                            flip=False):
        """
        Mask out the out-of-transit light curve based on transit parameters

        Parameters
        ----------
        params : `~batman.TransitParams`
            Transit light curve parameters. Requires that `params.duration`
            is defined.
        oot_duration_fraction : float (optional)
            Fluxes from what fraction of a transit duration of the
            out-of-transit light curve should be included in the mask?
        flip : bool (optional)
            If `True`, mask in-transit rather than out-of-transit.

        Returns
        -------
        d : dict
            Inputs for a new `LightCurve` object with the mask applied.
        """
        # Fraction of one duration to capture out of transit

        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + oot_duration_fraction)) |
                        (phased > params.per - params.duration*(0.5 + oot_duration_fraction)))
        if flip:
            near_transit = ~near_transit
        sort_by_time = np.argsort(self.times[near_transit].jd)
        return dict(times=self.times[near_transit][sort_by_time],
                    fluxes=self.fluxes[near_transit][sort_by_time],
                    errors=self.errors[near_transit][sort_by_time],
                    quarters=self.quarters[near_transit][sort_by_time])

    def mask_in_transit(self, params, oot_duration_fraction=0.25):
        """
        Mask out the in-transit light curve based on transit parameters

        Parameters
        ----------
        params : `~batman.TransitParams`
            Transit light curve parameters. Requires that `params.duration`
            is defined.
        oot_duration_fraction : float (optional)
            Fluxes from what fraction of a transit duration of the
            out-of-transit light curve should be included in the mask?

        Returns
        -------
        d : dict
            Inputs for a new `LightCurve` object with the mask applied.
        """
        return self.mask_out_of_transit(params, flip=True,
                                        oot_duration_fraction=oot_duration_fraction)

    def get_transit_light_curves(self, params, plots=False):
        """
        For a light curve with transits only (i.e. like one returned by
        `LightCurve.mask_out_of_transit`), split up the transits into their
        own light curves, return a list of `TransitLightCurve` objects.

        Parameters
        ----------
        params : `~batman.TransitParams`
            Transit light curve parameters

        plots : bool
            Make diagnostic plots.

        Returns
        -------
        transit_light_curves : list
            List of `TransitLightCurve` objects
        """
        time_diffs = np.diff(sorted(self.times.jd))
        diff_between_transits = params.per/2.
        split_inds = np.argwhere(time_diffs > diff_between_transits) + 1

        if len(split_inds) > 0:

            split_ind_pairs = [[0, split_inds[0][0]]]
            split_ind_pairs.extend([[split_inds[i][0], split_inds[i+1][0]]
                                     for i in range(len(split_inds)-1)])
            split_ind_pairs.extend([[split_inds[-1], len(self.times)]])

            transit_light_curves = []
            counter = -1
            for start_ind, end_ind in split_ind_pairs:
                counter += 1
                if plots:
                    plt.plot(self.times.jd[start_ind:end_ind],
                             self.fluxes[start_ind:end_ind], '.-')
                #print(start_ind, end_ind)
                if type(start_ind) is list or type(start_ind) is np.ndarray:
                    start_ind = start_ind[0]
                parameters = dict(times=self.times[start_ind:end_ind],
                                  fluxes=self.fluxes[start_ind:end_ind],
                                  errors=self.errors[start_ind:end_ind],
                                  quarters=self.quarters[start_ind:end_ind],
                                  name=counter)
                transit_light_curves.append(TransitLightCurve(**parameters))
            if plots:
                plt.show()
        else:
            transit_light_curves = []

        return transit_light_curves

    def get_available_quarters(self):
        """
        Get which quarters are available in this `LightCurve`

        Returns
        -------
        qs : list
            List of unique quarters available.
        """
        return list(set(self.quarters))

    def get_quarter(self, quarter):
        """
        Get a copy of the data from within `LightCurve` during one Kepler
        quarter.

        Parameters
        ----------
        quarter : int
            Kepler Quarter

        Returns
        -------
        lc : `LightCurve`
            Light curve from one Kepler Quarter
        """
        this_quarter = self.quarters == quarter
        return LightCurve(times=self.times[this_quarter],
                          fluxes=self.fluxes[this_quarter],
                          errors=self.errors[this_quarter],
                          quarters=self.quarters[this_quarter],
                          name=self.name + '_quarter_{0}'.format(quarter))

    @property
    def times_jd(self):
        """
        Get the times in this light curve in JD.

        Returns
        -------
        t_jd : `~numpy.ndarray`
            Julian dates.
        """
        return self.times.jd

    def split_at_index(self, index):
        """
        Split the light curve into two light curves, at ``index``
        """
        return (LightCurve(times=self.times[:index], fluxes=self.fluxes[:index], 
                           errors=self.errors[:index], quarters=self.quarters[:index], 
                           name=self.name),
                LightCurve(times=self.times[index:], fluxes=self.fluxes[index:], 
                           errors=self.errors[index:], quarters=self.quarters[index:], 
                           name=self.name))

    def transit_model(self, transit_params, short_cadence=True):
        # (1 * u.min).to(u.day).value
        if short_cadence:
            exp_time = (1 * u.min).to(u.day).value #(6.019802903 * 10 * u.s).to(u.day).value
            supersample = 10
        else:
            exp_time = (6.019802903 * 10 * 30 * u.s).to(u.day).value
            supersample = 10

        m = batman.TransitModel(transit_params, self.times.jd,
                                supersample_factor=supersample,
                                exp_time=exp_time)
        model_flux = m.light_curve(transit_params)
        return model_flux


class TransitLightCurve(LightCurve):
    """
    Container for a single transit light curve. Subclass of `LightCurve`.
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None,
                 name=None):
        """
        Parameters
        ----------
        times : `~numpy.ndarray`
            Times in JD
        fluxes : `~numpy.ndarray`
            Fluxes (normalized or not)
        errors : `~numpy.ndarray`
            Uncertainties on the fluxes
        quarters : `~numpy.ndarray` (optional)
            Kepler Quarter for each flux
        name : str
            Name this light curve (optional)
        """

        if isinstance(times[0], Time) and isinstance(times, np.ndarray):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')
        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        if self.times is not None and quarters is None:
            quarters = np.zeros_like(self.fluxes) - 1
        self.quarters = quarters
        self.name = name
        self.rescaled = False

    def fit_linear_baseline(self, params, cadence=1*u.min,
                            return_near_transit=False, plots=False):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT.

        Parameters
        ----------
        params : `~batman.TransitParams`
            Transit light curve parameters. Requires that `params.duration`
            is defined.
        cadence : `~astropy.units.Quantity` (optional)
            Length of the exposure time for each flux. Default is 1 min.
        return_near_transit : bool (optional)
            Return the mask for times in-transit.

        Returns
        -------
        linear_baseline : `numpy.ndarray`
            Baseline trend of out-of-transit fluxes
        near_transit : `numpy.ndarray` (optional)
            The mask for times in-transit.
        """
        cadence_buffer = cadence.to(u.day).value
        get_oot_duration_fraction = 0
        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration *
                         (0.5 + get_oot_duration_fraction) + cadence_buffer) |
                        (phased > params.per - params.duration *
                         (0.5 + get_oot_duration_fraction) - cadence_buffer))

        # Remove linear baseline trend
        order = 1
        linear_baseline = np.polyfit(self.times.jd[~near_transit],
                                     self.fluxes[~near_transit], order)
        linear_baseline_fit = np.polyval(linear_baseline, self.times.jd)

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, linear_baseline_fit, 'r')
            ax[0].plot(self.times.jd, self.fluxes, 'bo')
            plt.show()

        if return_near_transit:
            return linear_baseline, near_transit
        else:
            return linear_baseline

    def remove_linear_baseline(self, params, plots=False, cadence=1*u.min):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT,
        divide whole light curve by that fit.

        Parameters
        ----------
        params : `~batman.TransitParams`
            Transit light curve parameters. Requires that `params.duration`
            is defined.
        cadence : `~astropy.units.Quantity` (optional)
            Length of the exposure time for each flux. Default is 1 min.
        plots : bool (optional)
            Show diagnostic plots.
        """

        linear_baseline, near_transit = self.fit_linear_baseline(params,
                                                                 cadence=cadence,
                                                                 return_near_transit=True)
        linear_baseline_fit = np.polyval(linear_baseline, self.times.jd)
        self.fluxes =  self.fluxes/linear_baseline_fit
        self.errors = self.errors/linear_baseline_fit

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, self.fluxes, 'o')
            ax[0].set_title('before trend removal')

            ax[1].set_title('after trend removal')
            ax[1].axhline(1, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()

    def scale_by_baseline(self, linear_baseline_params):
        if not self.rescaled:
            scaling_vector = np.polyval(linear_baseline_params, self.times.jd)
            self.fluxes *= scaling_vector
            self.errors *= scaling_vector
            self.rescaled = True


    def fit_polynomial_baseline(self, params, order=2, cadence=1*u.min,
                                plots=False, mask=None):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT
        """
        if mask is None:
            mask = np.ones(len(self.fluxes)).astype(bool)
        cadence_buffer = cadence.to(u.day).value
        get_oot_duration_fraction = 0
        phased = (self.times.jd[mask] - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + get_oot_duration_fraction) + cadence_buffer) |
                        (phased > params.per - params.duration*(0.5 + get_oot_duration_fraction) - cadence_buffer))

        # Remove polynomial baseline trend after subtracting the times by its
        # mean -- this improves numerical stability for polyfit
        downscaled_times = self.times.jd - self.times.jd.mean()
        polynomial_baseline = np.polyfit(downscaled_times[mask][~near_transit],
                                         self.fluxes[mask][~near_transit], order)
        polynomial_baseline_fit = np.polyval(polynomial_baseline, downscaled_times)

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, polynomial_baseline_fit, 'r')
            ax[0].plot(self.times.jd, self.fluxes, 'bo')
            if mask is not None:
                ax[0].plot(self.times.jd[~mask], self.fluxes[~mask], 'ro')
            plt.show()

        return polynomial_baseline_fit

    def subtract_polynomial_baseline(self, params, plots=False, order=2,
                                     cadence=1*u.min):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit polynomial baseline to OOT,
        subtract whole light curve by that fit.
        """

        polynomial_baseline_fit = self.fit_polynomial_baseline(cadence=cadence,
                                                               order=order,
                                                               params=params)
        self.fluxes = self.fluxes - polynomial_baseline_fit
        self.errors = self.errors

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, self.fluxes, 'o')
            #ax[0].plot(self.times.jd[near_transit], self.fluxes[near_transit], 'ro')
            ax[0].set_title('before trend removal')

            ax[1].set_title('after trend removal')
            ax[1].axhline(1, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()

    def remove_polynomial_baseline(self, params, plots=False, order=2,
                                   cadence=1*u.min):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit polynomial baseline to OOT,
        divide whole light curve by that fit.
        """

        polynomial_baseline_fit = self.fit_polynomial_baseline(cadence=cadence,
                                                               order=order,
                                                               params=params)
        self.fluxes = self.fluxes / polynomial_baseline_fit
        self.errors = self.errors

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, self.fluxes, 'o')
            #ax[0].plot(self.times.jd[near_transit], self.fluxes[near_transit], 'ro')
            ax[0].set_title('before trend removal')

            ax[1].set_title('after trend removal')
            ax[1].axhline(1, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()



    def subtract_add_divide_without_outliers(self, params, quarterly_max,
                                             order=2, cadence=1*u.min,
                                             outlier_error_multiplier=50,
                                             outlier_tolerance_depth_factor=0.20,
                                             plots=False):

        init_baseline_fit = self.fit_polynomial_baseline(order=order,
                                                         cadence=cadence,
                                                         params=params)

        # Subtract out a transit model
        transit_model = generate_lc_depth(self.times_jd, params.rp**2, params)

        lower_outliers = (transit_model*init_baseline_fit - self.fluxes >
                          self.fluxes.mean() * outlier_tolerance_depth_factor *
                          params.rp**2)

        self.errors[lower_outliers] *= outlier_error_multiplier

        final_baseline_fit = self.fit_polynomial_baseline(order=order,
                                                          cadence=cadence,
                                                          params=params,
                                                          mask=~lower_outliers)

        self.fluxes = self.fluxes - final_baseline_fit
        self.fluxes += quarterly_max
        self.fluxes /= quarterly_max
        self.errors /= quarterly_max

        if plots:
            plt.errorbar(self.times.jd, self.fluxes, self.errors, fmt='o')
            plt.plot(self.times.jd[lower_outliers],
                     self.fluxes[lower_outliers], 'rx')
            plt.show()

    @classmethod
    def from_dir(cls, path):
        """Load light curve from numpy save files in ``path``"""
        times, fluxes, errors, quarters = [np.loadtxt(os.path.join(path, '{0}.txt'.format(attr)))
                                           for attr in ['times_jd', 'fluxes', 'errors', 'quarters']]

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path
        return cls(times, fluxes, errors, quarters=quarters, name=name)


def concatenate_transit_light_curves(light_curve_list, name=None):
    """
    Combine multiple transit light curves into one `TransitLightCurve` object.

    Parameters
    ----------
    light_curve_list : list
        List of `TransitLightCurve` objects
    name : str
        Name of new light curve

    Returns
    -------
    tlc : `TransitLightCurve`
        Concatenated transit light curves
    """
    times = []
    fluxes = []
    errors = []
    quarters = []
    for light_curve in light_curve_list:
        times.append(light_curve.times.jd)
        fluxes.append(light_curve.fluxes)
        errors.append(light_curve.errors)
        quarters.append(light_curve.quarters)
    times, fluxes, errors, quarters = [np.concatenate(i)
                                       for i in [times, fluxes,
                                                 errors, quarters]]

    times = Time(times, format='jd')
    return TransitLightCurve(times=times, fluxes=fluxes, errors=errors,
                             quarters=quarters, name=name)

def concatenate_light_curves(light_curve_list, name=None):
    """
    Combine multiple transit light curves into one `TransitLightCurve` object.

    Parameters
    ----------
    light_curve_list : list
        List of `TransitLightCurve` objects
    name : str
        Name of new light curve

    Returns
    -------
    tlc : `TransitLightCurve`
        Concatenated transit light curves
    """
    times = []
    fluxes = []
    errors = []
    quarters = []
    for light_curve in light_curve_list:
        times.append(light_curve.times.jd)
        fluxes.append(light_curve.fluxes)
        errors.append(light_curve.errors)
        quarters.append(light_curve.quarters)
    times, fluxes, errors, quarters = [np.concatenate(i)
                                       for i in [times, fluxes,
                                                 errors, quarters]]

    times = Time(times, format='jd')
    return LightCurve(times=times, fluxes=fluxes, errors=errors,
                      quarters=quarters, name=name)

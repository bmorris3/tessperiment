# Licensed under the MIT License - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import datetime
import os, subprocess, shutil, time
import numpy as np
from astropy.io import ascii
from .lightcurve import LightCurve

from threading import Lock

lock = Lock()

stsp_executable = os.path.abspath('/Users/bmmorris/git/STSP/stsp_20170714') # 07

infile_template_l = """#PLANET PROPERTIES
1							; Number of planets -- (if there are more than 1 planet, then the set of 8 planet properties are repeated)
{t0:2.10f}					; T0, epoch         (middle of first transit) in days.
{period:2.10f}				; Planet Period      (days)
{depth:2.10f}				; (Rp/Rs)^2         (Rplanet / Rstar )^ 2
{duration:2.10f}			; Duration (days)   (physical duration of transit, not used)
{b:2.10f}					; Impact parameter  (0= planet cross over equator)
{inclination:2.10f}			; Inclination angle of orbit (90 deg = planet crosses over equator)
{lam:2.10f}					; Lambda of orbit (0 deg = orbital axis along z-axis)
{ecosw:2.10f}			; ecosw
{esinw:2.10f}			; esinw
#STAR PROPERTIES
{rho_s:2.10f} 			; Mean Stellar density (Msun/Rsun^3)
{per_rot:2.10f}			; Stellar Rotation period (days)
4780					; Stellar Temperature
0.31					; Stellar metallicity
{tilt_from_z:2.10f}						; Tilt of the rotation axis of the star down from z-axis (degrees)
{nonlinear_ld}			; Limb darkening (4 coefficients)
{n_ld_rings:d}			; number of rings for limb darkening appoximation
#SPOT PROPERTIES
{n_spots}						; number of spots
0.7					; fractional lightness of spots (0.0=total dark, 1.0=same as star)
#LIGHT CURVE
{model_path}			; lightcurve input data file
{start_time:2.10f}		; start time to start fitting the light curve
{lc_duration:2.10f}		; duration of light curve to fit (days)
{real_max:2.10f}		; real maximum of light curve data (corrected for noise), 0 -> use downfrommax
1						; is light curve flattened (to zero) outside of transits?
#ACTION
l						; l= generate light curve from parameters
{spot_params}
1.00
"""

spot_params_template = """{spot_radius:2.10f}		; spot radius
{spot_theta:2.10f}		; theta
{spot_phi:2.10f}		; phi
"""

def quadratic_to_nonlinear(u1, u2):
    a1 = a3 = 0
    a2 = u1 + 2*u2
    a4 = -u2
    return (a1, a2, a3, a4)


def rho_star(transit_params):
    import astropy.units as u
    from astropy.constants import G, M_sun, R_sun
    """Calculate stellar density from MCMC chain samples"""
    #
    # aRs, i_deg = T14b2aRsi(transit_params.per, transit_params.duration,
    #                        transit_params.b, transit_params.rp,
    #                        transit_params.ecc, transit_params.w)
    aRs = transit_params.a

    rho_s = 3*np.pi/(G*(transit_params.per*u.day)**2) * aRs**3
    rho_s = rho_s.to(M_sun/(4./3 * np.pi * R_sun**3))
    return rho_s.value


def clean_up(require_input=False):
    paths_to_clean = glob(os.path.abspath(os.path.join(os.path.dirname(__file__), '.friedrich_tmp_*')))
    if require_input:
        user_input = input("Delete following paths [y]/n: \n" +
                           '\n'.join(paths_to_clean))
        if not user_input.lower() == 'n':
            for directory in paths_to_clean:
                shutil.rmtree(directory)
    else:
        for directory in paths_to_clean:
            shutil.rmtree(directory)

class STSP(object):
    def __init__(self, lc, transit_params, spot_params, outdir=None, keep_dir=False):
        """
        Parameters
        ----------
        lc : `friedrich.lightcurve.LightCurve`
            Light curve object
        transit_params : `batman.TransitParams`
            Parameters for planet and star
        spot_params : `numpy.ndarray`
            [r, theta, phi] for each spot to model with STSP
        outdir : str
            Directory to write temporary outputs into
        """
        self.lc = lc
        self.transit_params = transit_params
        self.spot_params = np.array(spot_params)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_integer = np.random.randint(0, 1e6)

        if outdir is None:
            self.outdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '.friedrich_tmp_{0}_{1}'.format(current_time, random_integer)))
        else:
            self.outdir = outdir

        #if not os.path.exists(self.outdir):
        os.makedirs(self.outdir)

        self.model_path = os.path.join(self.outdir, 'model_lc.dat')
        self.keep_dir = keep_dir

    def __enter__(self):
        return self

    def __exit__(self, *args):
        #self.safe_clean_up()
        if not self.keep_dir:
            shutil.rmtree(self.outdir)
            #clean_up()

    def safe_clean_up(self):
        paths_to_delete = ['model_lc.dat', 'test.in', 'xyzdetail.txt',
                           'test_lcout.txt', 'test_errstsp.txt']
        for path in paths_to_delete:
            abspath = os.path.join(self.outdir, path)
            if os.path.exists(abspath):
                os.remove(abspath)

    def stsp_lc(self, n_ld_rings=40, verbose=False, t_bypass=False, stsp_exec=None):
        #self.safe_clean_up()

        if stsp_exec is None:
            stsp_exec = stsp_executable

        # Normalize light curve to unity
        real_max = 1

        t_buffer = 0.08
        n_transits = np.rint(np.median((self.transit_params.t0 -
                                        self.lc.times.jd) /
                                       self.transit_params.per))
        if not t_bypass: 
            times = self.lc.times.jd + n_transits*self.transit_params.per
        else: 
            times = self.lc.times.jd
        fluxes = np.ones_like(times)

        np.savetxt(self.model_path,
                   np.vstack([times, fluxes,
                              fluxes]).T,
                   fmt=str('%1.10f'), delimiter='\t', header='stspinputs')

        # Calculate parameters for STSP:
        eccentricity, omega = self.transit_params.ecc, self.transit_params.w
        ecosw = eccentricity*np.cos(np.radians(omega))
        esinw = eccentricity*np.sin(np.radians(omega))
        start_time = times[0]#self.lc.times.jd[0]
        lc_duration = times[-1] - times[0]#self.lc.times.jd[-1] - self.lc.times.jd[0]
        nonlinear_ld = quadratic_to_nonlinear(*self.transit_params.u)
        nonlinear_ld_string = ' '.join(map("{0:.5f}".format, nonlinear_ld))

        # get spot parameters sorted out
        spot_params_str = spot_params_to_string(self.spot_params)

        # Stick those values into the template file

        in_file_text = infile_template_l.format(period=self.transit_params.per,
                                              ecosw=ecosw,
                                              esinw=esinw,
                                              lam=self.transit_params.lam,
                                              tilt_from_z=90-self.transit_params.inc_stellar,
                                              start_time=start_time,
                                              lc_duration=lc_duration,
                                              real_max=real_max,
                                              per_rot=self.transit_params.per_rot,
                                              rho_s=rho_star(self.transit_params),
                                              depth=self.transit_params.rp**2,
                                              duration=self.transit_params.duration,
                                              t0=self.transit_params.t0,
                                              b=self.transit_params.b,
                                              inclination=self.transit_params.inc,
                                              nonlinear_ld=nonlinear_ld_string,
                                              n_ld_rings=n_ld_rings,
                                              spot_params=spot_params_str[:-1],
                                              n_spots=int(len(self.spot_params)/3),
                                              model_path=os.path.basename(self.model_path))

        # Write out the `.in` file
        with open(os.path.join(self.outdir, 'test.in'), 'w') as in_file:
            in_file.write(in_file_text)

        # Run STSP
        # old_cwd = os.getcwd()
        # os.chdir(self.outdir)

        # stdout = subprocess.check_output(stsp_exec + ' test.in',
        #                                  #cwd=self.outdir,
        #                                  shell=True)
        # print(stdout)


        # if not verbose:
        # stdout = subprocess.check_output([stsp_exec, 'test.in'],
        #                                  #cwd=self.outdir,
        #                                  shell=True)


        # subprocess.check_call([stsp_exec, 'test.in'],
        #                        cwd=self.outdir, shell=True)
        #if verbose:
        #    print(stdout)

        #     subprocess.check_call([stsp_exec, 'test.in'])
        # else:
        #     stdout = subprocess.check_output([stsp_exec, 'test.in'])
        #     print(stdout.decode('ascii'))
        try:
            #subprocess.check_output([stsp_exec, 'test.in'], cwd=self.outdir)
            stdout = subprocess.check_output([stsp_exec, 'test.in'], cwd=self.outdir)
        except subprocess.CalledProcessError as err:
            pass#print("Failed. Error:", err.output, err.stderr, err.stdout)


        # os.chdir(old_cwd)
        time.sleep(0.01)
        # Read the outputs
        if os.stat(os.path.join(self.outdir, 'test_lcout.txt')).st_size == 0:
            stsp_times = self.lc.times.jd
            stsp_fluxes = np.ones_like(self.lc.fluxes)
            stsp_flag = 0 * np.ones_like(self.lc.fluxes)

        else:
            tbl = ascii.read(os.path.join(self.outdir, 'test_lcout.txt'), format='fast_no_header')
            stsp_times, stsp_fluxes, stsp_flag = tbl['col1'], tbl['col4'], tbl['col5']
        return LightCurve(times=stsp_times, fluxes=stsp_fluxes, quarters=stsp_flag)

def spot_params_to_string(spot_params):
    spot_params_str = ""
    for param_set in np.split(spot_params, len(spot_params)/3):
        spot_params_str += spot_params_template.format(spot_radius=param_set[0],
                                                       spot_theta=param_set[1],
                                                       spot_phi=param_set[2])
    return spot_params_str



# def friedrich_results_to_stsp_inputs(results_dir, transit_params):
#     """
#     Take outputs from friedrich, turn them into STSP inputs.
#     """
#
#     chains_paths = sorted(glob(os.path.join(results_dir, 'chains???.hdf5')))
#
#     for path in chains_paths:
#         m = MCMCResults(path, transit_params)
#         thetas, phis = m.max_lnp_theta_phi_stsp()
#
#         phis[phis < 0] += 2*np.pi
#         if len(thetas) > 1:
#
#             def spot_model(radii, mcmc, thetas=thetas, phis=phis):
#                 if len(thetas) > 1:
#                     spot_params = []
#                     for r, t, p in zip(radii, thetas, phis):
#                         spot_params.extend([r, t, p])
#                 else:
#                     spot_params = [radii[0], thetas[0], phis[0]]
#
#
#                 s = STSP(mcmc.lc, mcmc.transit_params, spot_params)
#                 t_model, f_model = s.stsp_lc()
#                 return t_model, f_model
#
#             def spot_chi2(radii, mcmc=m):
#                 t_model, f_model = spot_model(radii, mcmc=mcmc)
#
#                 first_ind = 0
#                 eps = 1e-5
#                 if np.abs(t_model.data[0] - mcmc.lc.times.jd[0]) > eps:
#                     for ind, time in enumerate(mcmc.lc.times.jd):
#                         if np.abs(t_model.data[0] - time) < eps:
#                             first_ind = ind
#                 chi2 = np.sum((mcmc.lc.fluxes[first_ind:] - f_model)**2 /
#                                mcmc.lc.errors[first_ind:]**2)
#                 return chi2
#
#             init_radii = np.zeros(len(thetas)) + 0.4 * m.transit_params.rp
#
#             from scipy.optimize import fmin
#             best_radii = fmin(spot_chi2, init_radii[:], xtol=1e-8)
#
#             if len(best_radii.shape) == 0:
#                 best_radii = [best_radii.tolist()]
#
#             init_t, init_f = spot_model(init_radii, m)
#             best_t, best_f = spot_model(best_radii, m)
#
#             spot_params_out = []
#             for r, t, p in zip(best_radii, thetas, phis):
#                 spot_params_out.extend([r, t, p])
#
#             stsp_params_out = spot_params_to_string(np.array(spot_params_out))
#
#             transit_number = int(m.index.split('chains')[1])
#
#             stsp_out_path = os.path.join(results_dir,
#                                          'stsp_spots{0:03d}.txt'.format(transit_number))
#             with open(stsp_out_path, 'w') as stsp_params_file:
#                 stsp_params_file.write(stsp_params_out)
#
#             fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
#             minjdint = int(np.min(m.lc.times.jd))
#             ax[0].errorbar(m.lc.times.jd - minjdint, m.lc.fluxes, m.lc.errors, color='k', fmt='.')
#             ax[0].plot(best_t - minjdint, init_f, 'g', lw=1)
#             ax[0].plot(best_t - minjdint, best_f, 'r', lw=2)
#             ax[0].set(ylabel='Flux',
#                        xlim=(np.min(m.lc.times.jd - minjdint),
#                              np.max(m.lc.times.jd - minjdint)),
#                        ylim=(0.995, 1.001),
#                        title='{0}'.format(m.index))
#             ax[1].set(xlabel='JD - {0}'.format(minjdint), ylabel='Residuals')
#
#             ax[1].plot(m.lc.times.jd - minjdint, m.lc.fluxes - best_f, 'k.')
#             ax[1].axhline(0, ls='--', color='r')
#             fig.tight_layout()
#             plt.savefig('tmp/{0}.png'.format(m.index), bbox_inches='tight')
#             plt.close()

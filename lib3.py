import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import rand
import pandas as pd
import scipy.special as sp
from matplotlib import cm
from scipy import integrate
from scipy.stats import norm
import concurrent.futures
from mpmath import spherharm
from scipy.special import gegenbauer as GG
from scipy.special import factorial2 as FF
from math import factorial

lmax = 2048
coeffSin = [((1j) ** l - (-1j) ** l) / (2j) for l in np.arange(lmax + 1)]
coeffCos = [((1j) ** l + (-1j) ** l) / (2) for l in np.arange(lmax + 1)]
PlaneWaveLMSin = lambda l, m, theta, phi: 4 * np.pi * coeffSin[l] * np.conj(spherharm(m, l, phi, theta))
PlaneWaveLMCos = lambda l, m, theta, phi: 4 * np.pi * coeffCos[l] * np.conj(spherharm(m, l, phi, theta))
PlaneWaveLM = PlaneWaveLMSin


HyperY = lambda k, l, m, ksi, theta, phi: np.sin(ksi)**l*sp.sph_harm(m, l, phi, theta)*np.sum(GG(1+l,k-l, np.cos(ksi)))


def plot_l_ortho_hyper(l, nside, k, ksi):
    def hypersphericalharm(k, l, m, ksi, theta, phi):
        N = (-1) ** k * (1j) ** l * FF(2 * l) * np.sqrt(2 * (k + 1) * factorial(k - l) / np.pi / factorial(k + l + 1))
        return N * HyperY(k, l, m, ksi, theta, phi)

    mm = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside=nside, ipix=mm)
    for m in range(0, l + 1):
        fig = plt.figure(figsize=[12, 12])
        fcolors = hypersphericalharm(k, l, m, ksi, theta, phi)
        fmin = np.min(fcolors)
        fmax = np.max(fcolors)
        fcolors = (fcolors - fmin) / (fmax - fmin)
        ax = hp.orthview(fcolors, min=-1, max=1, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()



def extractFColors(df, nside, lmax, sigma_smica, jj=None):
    fd = pd.DataFrame(df, columns=["l", "m", "alm"])
    ind = fd.l == fd.m
    fd.loc[ind, "alm"] = fd.loc[ind, "alm"] * jj
    Z = np.array(fd.alm.values.tolist(), dtype=np.complex)
    fcolors = hp.alm2map(Z, nside, lmax)
    (mu, sigma) = norm.fit(fcolors)
    fcolors = sigma_smica / sigma * (fcolors - mu)
    return fcolors


def create_eikxyz(thetaPrime, phiPrime, lmax):
    alm_map_all = {}
    dimension = len(thetaPrime)
    for i in np.arange(dimension):
        alm_map_all[i] = create_eikx_alm_map(thetaPrime[i], phiPrime[i], lmax)
    return alm_map_all


def Hyper_LM_Decomposition(k, l, m, lambda_0, lambda_k, lambda_l, lambda_m):
    return np.exp(-l / lambda_0) * (np.sin(2 * np.pi * l / lambda_l) / (2 * np.pi * l)) * (
                np.sin(2 * np.pi * l / lambda_m) / (2 * np.pi * l))


def create_hyper_sph_alm_map_long(thetaphase, lmax, lambda_0, lambda_l, lambda_m, nside, sigma_smica):
    alm_map = []
    for l in np.arange(lmax + 1):
        for m in np.arange(0, l + 1):
            x = l * (2 * np.pi + thetaphase) + 1E-3
            alm = Hyper_LM_Decomposition(x, m, lambda_0, lambda_l, lambda_m)
            alm_map.append([l, m, alm])
    fd = pd.DataFrame(alm_map, columns=["l", "m", "alm"])
    Z = np.array(fd.alm.values.tolist(), dtype=np.complex)
    fcolors = hp.alm2map(Z, nside, lmax)
    (mu, sigma) = norm.fit(fcolors)
    fcolors = sigma_smica / sigma * (fcolors - mu)
    return fcolors, fd

def create_eikx_alm_map_long(thetaPrime, phiPrime, lmax):
    alm_map = []
    for l in np.arange(lmax + 1):
        for m in np.arange(0, l + 1):
            alm = PlaneWaveLM(l, m, thetaPrime[0], phiPrime[0]) / np.sqrt(2 * np.pi)
            alm += PlaneWaveLM(l, m, thetaPrime[1], phiPrime[1]) / np.sqrt(2 * np.pi)
            alm += PlaneWaveLM(l, m, thetaPrime[2], phiPrime[2]) / np.sqrt(2 * np.pi)
            alm_map.append([l, m, alm])
    return alm_map


def create_eikx_alm_map(thetaPrime, phiPrime, lmax):
    alm_map = []
    for l in np.arange(lmax + 1):
        for m in np.arange(0, l + 1):
            if l != m:
                alm = 0.0 + 0.0j
            else:
                alm = PlaneWaveLM(l, m, thetaPrime[0], phiPrime[0]) / np.sqrt(2 * np.pi)
                alm += PlaneWaveLM(l, m, thetaPrime[1], phiPrime[1]) / np.sqrt(2 * np.pi)
                alm += PlaneWaveLM(l, m, thetaPrime[2], phiPrime[2]) / np.sqrt(2 * np.pi)
            alm_map.append([l, m, alm])
    return alm_map


def eikx_alm_map(df, Cb, lmax, nside, planck_theory_cl, sigma_smica):
    for l in np.arange(lmax + 1):
        cl = 0.0
        alm_map = []
        for m in np.arange(0, l + 1):
            if l != m:
                alm = 0.0 + 0.0j
            else:
                alm = df[l] * Cb[l]
            alm_map.append([l, m, alm])
    alm_map = np.array(alm_map)
    fcolors = hp.alm2map(alm_map, nside, lmax=lmax)
    (mu, sigma) = norm.fit(fcolors)
    fcolors = sigma_smica / sigma * fcolors
    theta, phi, x, y, z = getSphericalXYZ(nside)
    plotSMICAHistogram(fcolors)
    fig = plotSMICA_aitoff(fcolors)
    plot_CL_From_Image(fcolors, planck_theory_cl)
    return alm_map


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolateNaNs(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def createRandomCL_Map(n=10):
    cl_map = []
    for l in range(n):
        for m in range(l):
            ampl = rand()
            a_theta = rand() * np.pi
            a_phi = rand() * np.pi
            cl_map.append([l, m, a_theta, a_phi, ampl])
    return cl_map


def createSphere(nside, cl_map):
    # sample spacing
    theta, phi, x, y, z = getSphericalXYZ(nside=nside)
    fcolors = np.zeros(len(theta)) + (1j) * np.zeros(len(theta))
    for l, m, clm in cl_map:
        fcolors += clm * norm_harmonic(l, m, theta, phi)
    fcolors = np.abs(fcolors)
    fmin = np.min(fcolors)
    fmax = np.max(fcolors)
    fcolors = (fcolors - fmin) / (fmax - fmin)
    return fcolors, x, y, z


def dotprod(f, g):
    # Scipy does not directly integrates complex functions.
    # You have to break them down into two integrals of the real and imaginary part
    integrand_r = lambda theta, phi: np.real(f(theta, phi) * np.conj(g(theta, phi)) * np.sin(theta))
    integrand_i = lambda theta, phi: np.imag(f(theta, phi) * np.conj(g(theta, phi)) * np.sin(theta))
    rr = integrate.dblquad(integrand_r, 0, 2 * np.pi, lambda theta: 0, lambda theta: np.pi)[0]
    ri = integrate.dblquad(integrand_i, 0, 2 * np.pi, lambda theta: 0, lambda theta: np.pi)[0]
    if np.allclose(rr, 0):
        rr = 0
    if np.allclose(ri, 0):
        ri = 0
    return rr + ri * 1j


def norm_harmonic(l, m, theta, phi):
    fcolors = sp.sph_harm(m, l, theta, phi).real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin) / (fmax - fmin)
    return fcolors


# def plot_harmonics(fcolors, x, y, z):
#     fig = plt.figure(figsize=plt.figaspect(1.))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(x, y, z, rstride=1, cstride=1, facecolors=cm.RdBu_r(fcolors))
#     ax.set_axis_off()
#     plt.show()


def plot_ax(fig, fcolors, x, y, z, numplots, pos):
    ax = fig.add_subplot(1, numplots, pos, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.RdBu_r(fcolors))
    ax.set_axis_off()
    return ax


def plot_l(l):
    theta = np.linspace(0, np.pi, 200)
    phi = np.linspace(0, 2 * np.pi, 200)
    theta, phi = np.meshgrid(theta, phi)
    Y = np.sin(theta) * np.sin(phi)
    X = np.sin(theta) * np.cos(phi)
    Z = np.cos(theta)
    fig = plt.figure(figsize=[12, 12])
    for m in range(0, l + 1):
        ax = plot_ax(fig, norm_harmonic(l, m, theta, phi), X, Y, Z, l + 1, m + 1)
    plt.show()
    return ax


def plot_harmonics(fcolors, x, y, z):
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, cmap=fcolors)
    ax.set_axis_off()
    plt.show()


def plot_l_ortho(l, nside):
    mm = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside=nside, ipix=mm)
    for m in range(0, l + 1):
        fig = plt.figure(figsize=[12, 12])
        fcolors = norm_harmonic(l, m, theta, phi)
        ax = hp.orthview(fcolors, min=-1, max=1, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()


def B_l(beam_arcmin, ls):
    theta_fwhm = ((beam_arcmin / 60.0) / 180) * math.pi
    theta_s = theta_fwhm / (math.sqrt(8 * math.log(2)))
    return np.exp(-2 * (ls + 0.5) ** 2 * (math.sin(theta_s / 2.0)) ** 2)


def getSphericalXYZ(nside=10):
    mm = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside=nside, ipix=mm)
    z = np.cos(theta)
    y = np.sin(theta) * np.sin(phi)
    x = np.sin(theta) * np.cos(phi)
    return theta, phi, x, y, z


def plotSMICA_aitoff(planck_IQU_SMICA):
    fig = plt.figure(1, figsize=[12, 12])
    hp.mollview(planck_IQU_SMICA, min=-0.0007, max=0.0007, title="Planck Temperature Map", fig=1, unit="K",
                cmap=cm.RdBu_r)
    hp.graticule()
    plt.show()
    return fig


def plotSMICAHistogram(fcolors):
    (mu, sigma) = norm.fit(fcolors)
    fig, ax = plt.subplots()
    n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y)
    plt.xlim(mu - 1 * sigma, mu + 1 * sigma)
    plt.xlabel("Temperature/K")
    plt.ylabel("Frequency")
    plt.title(r"Histogram of $12-N_{side}^2$ pixels from the Planck SMICA Map ", y=1.08)
    plt.show()


def plot_WhiteNoise(white_noise):
    fig = plt.figure()
    hp.mollview(white_noise, min=-0.0006, max=0.0006, title="White Noise Map", fig=1, unit=r'Temperature/K',
                cmap=cm.RdBu_r)
    plt.show()


def plotWhiteNoiseHistogram(white_noise):
    plt.hist(white_noise, bins=np.arange(-0.0005, 0.0005, 0.00002), color='b', alpha=0.2)
    plt.xlim(-0.0006, 0.0006)
    plt.xlabel("temperature/K")
    plt.ylabel('Frequency')
    plt.title("Histogram of $12N_(side)^2$ random sampels from a normal (Gaussian) distribution ")
    plt.show()


def plot_CL_From_Image(fcolors, planck_theory_cl, lmax=2048, xmax=1000):
    pl = hp.sphtfunc.pixwin(lmax)
    cl_SMICA = hp.anafast(fcolors, lmax=lmax)
    ell = np.arange(len(cl_SMICA))
    themax = np.max(planck_theory_cl[10:1000, 1])
    planck_theory_cl[:, 1] = planck_theory_cl[:, 1] / themax
    # Deconvolve the beam and the pixel window function
    dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl[0:(lmax + 1)] ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
    myMax = np.max(dl_SMICA[10:1000])
    dl_SMICA = dl_SMICA / myMax
    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111)
    ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
    ax.plot(planck_theory_cl[:, 0], planck_theory_cl[:, 1], ell * np.pi, dl_SMICA)
    ax.set_ylabel('$\ell$')
    ax.set_title("Angular Power Spectra")
    ax.legend(loc="upper right")
    #     ax.set_yscale("log")
    ax.set_xlim(10, xmax)
    plt.ylim(1E-10, 1)
    ax.grid()
    plt.show()
    return cl_SMICA, dl_SMICA, ell


def plot_CL_From_ALM(fcolors, lmax=2048):
    pl = hp.sphtfunc.pixwin(lmax)
    cl_SMICA = hp.anafast(fcolors, lmax=lmax)
    ell = np.arange(len(cl_SMICA))

    # Deconvolve the beam and the pixel window function
    dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl[0:1025] ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi)) / 1E-12
    myMax = np.max(dl_SMICA)
    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111)
    plt.plot(ell * np.pi, dl_SMICA / myMax)

    ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
    ax.set_ylabel('$\ell$')
    ax.set_title("Angular Power Spectra")
    ax.legend(loc="upper right")
    #     ax.set_yscale("log")
    ax.set_xlim(2, 1000)
    ax.set_ylim(1E-10, 1.0)
    ax.grid()
    plt.show()
    return cl_SMICA, dl_SMICA, ell


from requests import get
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from time import time
from random import sample
import numpy as np


def multithreading(function, iterable, number_of_threads):
    """
    Maps a function across an iterable (such as a list of elements) with the optional use of multithreading.

    :param function: name of a function
    :type function: function

    :param iterable: elements used as inputs to function parameter
    :type iterable: list

    :param number_of_threads: number of threads to use in map operation
    :type number_of_threads: int

    :returns list_objects: return objects from our function parameter calls
    :return type: list
    """
    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        responses = executor.map(function, iterable)
    return list(responses)


def do_multiprocessing(function, iterable, number_of_concurrent_processes):
    """
    Maps a function across an iterable (such as a list of elements) with the optional use of multiprocessing.

    :param function: name of a function
    :type function: function

    :param iterable: elements used as inputs to function parameter
    :type iterable: list

    :param number_of_concurrent_processes: number of concurrent processes in multiprocessing
    :type number_of_concurrent_processes: int

    :returns list_objects: return objects from our function parameter calls
    :return type: list
    """
    with ProcessPoolExecutor(max_workers=number_of_concurrent_processes) as executor:
        responses = executor.map(function, iterable)
    return list(responses)


def transform_timestamps_to_be_seconds_from_process_start_time(process_start_time, all_task_timestamps):
    """
    Take list of start and end timestamps of # of seconds since epoch, and subtract the process start time from them all

    Therefore we'll know how far timestamps are from the 0th second, the start of the program.

    :param process_start_time: # of seconds since epoch for start of task
    :type process_start_time: float

    :param all_task_timestamps: # of seconds since epoch for end of task
    :type all_task_timestamps: list

    :return function_timestamps_starting_from_zero: same shape as all_task_timestamps but all values subtracted by process_start_time
    :type function_timestamps_starting_from_zero: numpy array
    """
    function_timestamps_starting_from_zero = np.array(all_task_timestamps) - process_start_time
    return function_timestamps_starting_from_zero[0:2]


def separate_list_elements(list_of_lists):
    """
    Given a list structure such as [[x, y], [x, y]] return a list of just the x's and another of just y's

    :param list_of_lists: list with nested lists
    :type list_of_list: list

    :return start_values, end_values: two lists - one of all 0-th index values and another of 1st index values in each nested list
    :return type: tuple storing two lists
    """
    start_values = [inner_list[0] for inner_list in list_of_lists]
    start_values = np.array(start_values)

    end_values = [inner_list[1] for inner_list in list_of_lists]
    end_values = np.array(end_values)
    return start_values, end_values


def generate_bar_colors(number_of_threads_or_subprocesses):
    """
    Make a list of colors the same length as the number of threads or number of concurrent subprocesses

    :param number_of_threads_or_subprocesses: number of threads used in multithreading or number of processes used in multiprocessing
    :type number_of_threads_or_subprocesses: int

    :return colors: list of colors chosen from good_colors
    :type colors: list
    """
    good_colors = ['firebrick', 'darkgreen', 'royalblue', 'rebeccapurple', 'dimgrey', 'teal', 'chocolate',
                   'darkgoldenrod']
    colors = sample(good_colors, number_of_threads_or_subprocesses)
    return colors


def visualize_task_times(start_times, end_times, plot_title, colors):
    """
    Use Matplotlib module to create a horizontal bar chart of the time elapsed for each task.

    :param start_times: start times of tasks
    :type start_times: list

    :param end_times: end times of tasks
    :type end_times: list

    :param plot_title: title of plot
    :type plot_title: string

    :param colors: colors of bars
    :type colors: list

    :return: None
    """
    plt.barh(range(len(start_times)), end_times - start_times, left=start_times, color=colors)
    plt.grid(axis='x')
    plt.ylabel("Tasks")
    plt.xlabel("Seconds")
    plt.title(plot_title)
    plt.figure(figsize=(12, 10))
    plt.show()
    return None


def visualize_multiprocessing_effect(number_of_concurrent_processes, function_name, iterable, plot_title):
    """
    Perform multithreading given a function_name and number_of_threads and visualize tasks as bar chart

    :param number_of_concurrent_processes: number of concurrent processes in multiprocessing
    :type number_of_concurrent_processes: int

    :param function_name: name of function applied in multithreading operation
    :type function_name: function

    :param iterable: elements used as inputs to function parameter
    :type iterable: list

    :param plot_title: title of plot
    :type plot_title: string

    :return: None
    """
    process_start_time = time()  # we track time here
    time_logs_multiprocessing_op = do_multiprocessing(function_name, iterable, number_of_concurrent_processes)
    dt = [x[0:2] for x in time_logs_multiprocessing_op ]
    dr = [x[2:] for x in time_logs_multiprocessing_op ]
    multiprocessing_task_timestamps = transform_timestamps_to_be_seconds_from_process_start_time(process_start_time, dt)
    start_times, end_times = separate_list_elements(multiprocessing_task_timestamps)
    colors = generate_bar_colors(number_of_concurrent_processes)
    visualize_task_times(start_times, end_times, plot_title, colors)
    return dr

def visualize_multithreading_effect(number_of_threads, function_name, iterable, plot_title):
    """
    Perform multithreading given a function_name and number_of_threads and visualize tasks as bar chart

    :param number_of_threads: number of threads used in multithreading
    :type number_of_threads: int

    :param function_name: name of function applied in multithreading operation
    :type function_name: function

    :param iterable: elements used as inputs to function parameter
    :type iterable: list

    :param plot_title: title of plot
    :type plot_title: string

    :return: None
    """
    process_start_time = time()  # we track time here
    time_logs_multithreading_op = multithreading(function_name, iterable, number_of_threads)
    multithreading_task_timestamps = transform_timestamps_to_be_seconds_from_process_start_time(process_start_time,
                                                                                                time_logs_multithreading_op)
    start_times, end_times = separate_list_elements(multithreading_task_timestamps)
    colors = generate_bar_colors(number_of_threads)
    visualize_task_times(start_times, end_times, plot_title, colors)


def get_response_time_measurements(url):
    """
    mark start time, then call the get method and pass in a url to receive a server response object, then mark end time

    :param url: address of a worldwide web page
    :type url: string

    :returns: start_time and stop_time of this task
    :type returns: list
    """
    start_time = time()
    try:
        response = get(url)
    except Exception as exception_object:
        print('Error with request for url: {0}'.format(url))
    stop_time = time()
    return [start_time, stop_time]


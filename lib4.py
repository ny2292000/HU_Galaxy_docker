import operator
import os
import warnings
from os import listdir
from os.path import isfile, join

import matplotlib.pylab as plt
from PIL import Image
from numba import prange
from pyshtools.legendre import legendre
from scipy.optimize import minimize
from scipy.special import eval_gegenbauer

from lib3 import *
from parameters import *

warnings.filterwarnings("ignore")
import mpmath as mp
from enum import Enum
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.special import sici


def get_dl(fcolors, nside, beam_arc_min=5):
    cl_SMICA = hp.anafast(fcolors)
    ell = np.arange(len(cl_SMICA))
    pl = hp.sphtfunc.pixwin(nside=nside)
    dl_SMICA = cl_SMICA / (B_l(beam_arc_min, ell) ** 2 * pl ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
    return cl_SMICA, dl_SMICA, ell


class fitClass:
    def __init__(self):
        self.t = []
        self.smica = []
        self.white_noise = []
        self.func = self.correctWN
        self.n = 6

    def six_peaks(self, ell, *parkguess):
        'function of two overlapping peaks'
        p = np.zeros([1, len(ell)])
        # gamma = parkguess[-1:][0]
        for i in np.arange(0, self.n * 4, 4):
            a0 = parkguess[i + 0]  # peak height
            a1 = parkguess[i + 1]  # Gaussian Center
            a2 = parkguess[i + 2]  # std of gaussian
            gamma = parkguess[i + 3]  # std of gaussian
            p += self.fitfun(ell, a0, a1, a2, gamma)
        return p[0]

    def seven_peaks(self, x, *parkguess):
        'function of two overlapping peaks'
        p = np.zeros([1, len(x)])
        a0 = parkguess[0]  # first peak height
        a1 = parkguess[1]  # Gaussian Center
        a2 = parkguess[2]  # std of gaussian
        deltaL = parkguess[3] # shift in L
        gamma = parkguess[4]  # exponential rate of peak decay
        baseline = parkguess[5]
        for i in np.arange(self.n):
            p +=  a0 * norm.pdf(x, loc=a1+ deltaL*i, scale=a2) 
        return p[0] * np.exp(-gamma * x) + baseline
    
    
    
    def fitfun(self, x, a0, a1, a2, gamma):
        return a0 * norm.pdf(x, loc=a1, scale=a2) * np.exp(-gamma * x) * np.sqrt(2 * np.pi)

    def correctWN(self, x):
        return x[0] * np.exp(x[1] * self.t) * self.white_noise
        # return x[0]+x[1]*np.exp(x[2]*self.t)*self.white_noise

    def sins_quared(self, x):
        phase = self.t / x[3]
        data_1D = x[0] * sici(phase)[0]
        box_kernel = Gaussian1DKernel(x[1])
        smoothed_data_box = convolve(data_1D, box_kernel) * np.exp(x[2] * phase)
        return smoothed_data_box

    def fitGN(self, x):
        err = np.sum((self.smica - self.func(x)) ** 4 * 1E20)
        return err

    def ffitme(self,x0):
        a=x0[0]
        R=x0[1]
        xx = 2*np.pi*self.t/R
        return a*(np.sin(xx)-xx*np.cos(xx))/xx**3

    def optimizeme(self, x0):
        x00 = minimize(self.fitGN, x0,
                       method='nelder-mead', options={'xatol': 1e-18, 'disp': True, 'maxiter': 10000})
        err = x00.fun
        xx0 = x00.x
        return xx0, err

    def plotme(self, x0, ymax=None):
        plt.figure()
        plt.plot(self.t, self.smica)
        plt.plot(self.t, self.func(x0), 'r-')
        # plt.xlim([0, 2500])
        # plt.ylim([0, ymax])
        plt.show()

    def plotmeSix(self, x0):
        plt.plot(self.t, self.smica, self.t, self.func(x0))
        plt.xlim([0, None])
        plt.ylim([0, np.max(self.smica)])
        plt.show()
        
     
        
        


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)


def get_chopped_range(lmax, n=20, lmin=1):
    llist = sorted(list({int(np.round(x, 0)) for x in np.linspace(lmin, lmax, n)}))
    return llist


def sinxx(xx):
    y = np.zeros(xx.shape)
    for i, x in enumerate(xx):
        if x == 0:
            y[i] = 1.0
        else:
            y[i] = np.sin(x) / x
    return y


def optimzeSpectraC(df, x0, opt=False, kkk=None):
    def erroC(x, df):
        k = df.k - 2
        dk = x[0] * k + x[1] * k ** 2 + x[2] * k ** 3
        dl = x[3] * df.l + x[4] * df.l ** 2
        dm = x[5] * np.abs(df.m) + x[6] * df.m ** 2
        df["population"] = x[7] + dk + dl + dm
        dfG = pd.DataFrame(df.groupby(['k', "l"])["CL", "population"].apply(lambda x: x.astype(float).mean()), columns=["CL", "population"])
        dfG["CL"] /= dfG["CL"].max()
        err = np.sum((dfG.population - dfG.CL) ** 2)
        print(err)
        return err

    if opt:
        if kkk:
            df = df[(df.k == kkk[0]) * (df.l == kkk[1])]
        xout = minimize(erroC, x0, args=(df), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        print("errC", xout.fun, xout.x)
        return xout.x, df
    else:
        x = x0
    ##########################################################
    if kkk:
        df = df[(df.k == kkk[0]) * (df.l == kkk[1])]
        plt.plot(df.m, df.coeff)
        plt.show()
    k = df.k - 2
    dk = x[0] * k + x[1] * k ** 2 + x[2] * k ** 3
    dl = x[3] * df.l + x[4] * df.l ** 2
    dm = x[5] * np.abs(df.m) + x[6] * df.m ** 2
    df["population"] = x[7] + dk + dl + dm
    dfG = pd.DataFrame(df.groupby(['k', "l"])["CL", "population"].apply(lambda x: x.astype(float).mean()),
                   columns=["CL", "population"])
    dfG["CL"] /= dfG["CL"].max()
    dfG["difference"] = (dfG.CL - dfG.population) / dfG.population
    dfG["zero"] = 0.0
    dfG["one"] = 1.0
    ##########################################################
    ax = dfG.CL.plot(legend=False)
    # dfG.population.plot( ax=ax)

    plt.xlabel('Hyperspherical Harmonic k')
    plt.title('Power Spectrum C')
    plt.ylabel("Hyperspherical Harmonic Mode Population")
    # plt.ylim(0, 2)
    plt.show()


def plot_aitoff(fcolors, cmap=cm.RdBu_r):
    hp.mollview(fcolors.squeeze(), cmap=cmap)
    hp.graticule()
    plt.show()


def plot_aitoff_df(l, m, df, cmap=cm.RdBu_r):
    def plot_orth(fcolors):
        sigma = np.std(fcolors)
        hp.orthview(fcolors, min=-2 * sigma, max=2 * sigma, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()

    def get_LM(l, m, df):
        if m >= 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[0, ind, :]
        if m < 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[1, ind, :]

    fcolors = get_LM(l, m, df)
    plot_aitoff(fcolors, cmap=cmap)
    plot_orth(fcolors)


def B_l(beam_arcmin, ls1):
    theta_fwhm = ((beam_arcmin / 60.0) / 180) * math.pi
    theta_s = theta_fwhm / (math.sqrt(8 * math.log(2)))
    return np.exp(-2 * (ls1 + 0.5) ** 2 * (math.sin(theta_s / 2.0)) ** 2)


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def chunks(lst, n, args=None):
    """Yield successive n-sized chunks from lst."""
    maxn = len(lst)
    for i in range(0, maxn, n):
        nextstep = i + n
        if nextstep > maxn:
            nextstep = maxn
        if args is not None:
            yield (i, args, lst[i:nextstep])
        else:
            yield (i, lst[i:nextstep])


def chg2ang(alpha):
    residue = alpha % (2 * np.pi)
    return str(int(residue * 180 / np.pi))


def newerr(x, SMICA_LR, newmap):
    newnewmap = SMICA_LR.squeeze() - x[1] * newmap.squeeze() - x[0]
    err = np.var(newnewmap)
    return err


def olderr(x, dl_SMICA, ell):
    # err = np.std(dl_SMICA- x[0]*np.sin(x[1]*ell)**2/ell**2+x[2]*ell**x[3])
    err = dl_SMICA - x[0] - x[1] * np.exp(ell ** x[2]) - x[3] * np.sin(ell * x[4]) ** 2 / (x[4] * ell) ** 2
    err = np.sum(err * err)
    print(err, x)
    return err


def functionG(a):
    l = a[0][0] - 1
    k = a[0][1] + l
    cosksi = a[1]
    A = np.array([eval_gegenbauer(1 + l, k - l, x) for x in cosksi])
    #     print(a[0], "done")
    return a[0], A


class HYPER:
    def __init__(self, nside3D, sigma_smica, planck_IQU_SMICA, karray,mypath, 
                 lambda_k=0.0, lambda_l=0.0, lambda_m=0.0, bandwidth=256,
                 loadpriorG=False, savePG=False, longG=False):
        self.G = pd.DataFrame()
        self.loadpriorG = loadpriorG
        self.savePG = savePG
        self.extra_G_keys = {}
        self.extra_P_keys = {}
        self.longG = longG
        self.nside3D = nside3D
        self.bandwidth = bandwidth
        self.df = np.zeros([1, 1])
        self.results = {}
        self.sigma_smica = sigma_smica
        self.xx, self.yy, self.zz = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside3D)))
        self.theta3D, self.phi3D = hp.pix2ang(nside=nside3D, ipix=np.arange(hp.nside2npix(nside3D)))
        self.k1 = min(karray)
        self.kmax = max(karray)
        self.karray = karray
        self.mypath = mypath
        self.G_filename = os.path.join(self.mypath, "G_{}_{}_{}_{}.npy".format(self.nside3D, chg2ang(lambda_k), chg2ang(lambda_l), chg2ang(lambda_m)))
        self.P_filename = os.path.join(self.mypath, "P_{}_{}_{}_{}_".format(self.nside3D, chg2ang(lambda_k), chg2ang(lambda_l),chg2ang(lambda_m)))
        # self.G_filename = "./PG_data/G_64_492_495_709.npy"
        if loadpriorG:
            if os.path.exists(self.G_filename):
                print("loading from ", self.G_filename)
                self.G = pd.read_pickle(self.G_filename)
        self.sinksi = np.zeros([1, 1])
        self.cosksi = np.zeros([1, 1])
        self.phi = np.zeros([1, 1])
        self.x = np.zeros([1, 1])
        self.y = np.zeros([1, 1])
        self.z = np.zeros([1, 1])
        self.costheta = np.zeros([1, 1])
        self.lambda_k, self.lambda_l, self.lambda_m = (lambda_k, lambda_l, lambda_m)
        self.SMICA = self.normalizeFColors(planck_IQU_SMICA, sigma_smica)
        self.SMICA_LR = self.SMICA
        self.newmap = np.zeros(self.SMICA.shape)
        self.change_SMICA_resolution(nside3D, doit=True, bandwidth=bandwidth)
        self.change_HSH_center(self.lambda_k, self.lambda_l, self.lambda_m, self.karray,
                               self.nside3D, loadpriorG=loadpriorG, doit=True, savePG=savePG)

    def change_SMICA_resolution(self, nside3D, doit=False, bandwidth=256):
        if (self.nside3D == nside3D) and not doit:
            return
        filename = "./img1/SMICA_{}_{}.png".format(nside3D, bandwidth)
        title = "SMICA_nside3D={}_{}".format(nside3D, bandwidth)
        self.SMICA_LR = self.change_resolution(self.SMICA, nside3D=nside3D, bandwidth=bandwidth,
                                               title=title, filename=filename,
                                               plotme=True, save=True)

    def factorMP(self, k, l, m):
        m = np.abs(m)
        a = (-1) ** m * np.sqrt((2 * l + 1) / (2 * np.pi) * mp.factorial(l - m) / mp.factorial(l + m))
        b = (-1) ** k * np.sqrt(
            2 * (k + 1) / np.pi * mp.factorial(k - l) / mp.factorial(k + l + 1) * 2 ** (2 * l) * mp.factorial(l) ** 2)
        c = float(a * b)
        return c

    def normalizeFColors(self, fcolors, sigma_smica):
        (mu, sigma) = norm.fit(fcolors)
        if sigma == 0.0:
            return fcolors
        return sigma_smica / sigma * (fcolors - mu)

    def getSpectralFiles(self, mypath):
        return [join(mypath, f) for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]

    def creategiff(self, kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB"):
        giffiles = self.getSpectralFiles(mypath)
        gifnumbers = [x.replace(prefix,"").replace(".png", "").replace(mypath,"") for x in giffiles if prefix in x]
        gifnumbers = list(set(gifnumbers))
        gifnumbers = np.sort(gifnumbers)
        images = []
        for fff in gifnumbers:
            ff =  mypath + prefix + fff + ".png"
            print(ff)

        for fff in gifnumbers:
            ff =  mypath + prefix + fff + ".png"
            print(ff)
            with Image.open(ff).convert("RGB") as img:
                images.append(img)
        fname = os.path.join(mypath, filename) + '_{}_{}.gif'.format(kmax, nside3D)
        images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)
    

    def creategiffMem(self, x, kmax):
        # err, prime, results, fig
        images = [y[3] for y in x]
        titles = [y[1] for y in x]
        errs = [y[0] for y in x]
        for i, xx in enumerate(images):
            (lk, ll, lm) = [x for x in titles[i]]
            err = errs[i]
            filename = "./img1/aitoff_{}_{}_{}_{}.png".format(kmax, chg2ang(lk), chg2ang(ll), chg2ang(lm))
            # draw the canvas, cache the render
            xx.seek(0)
            im = Image.open(xx)
            im.save(filename)

    def change_resolution(self, fcolors, nside3D=1024, filename=None,
                          title=None, save=False, plotme=None, bandwidth=256):
        # noinspection PyUnresolvedReferences
        SMICA_alms = hp.map2alm(fcolors, lmax=bandwidth)
        fcolors = hp.alm2map(SMICA_alms, nside=nside3D)
        (mu, sigma) = norm.fit(fcolors)
        if sigma != 0.0:
            fcolors = (fcolors - mu) / sigma * self.sigma_smica
        fcolors = np.expand_dims(fcolors, axis=1)
        self.plot_aitoff(fcolors.squeeze(), kk=0, ll=0, mm=0, err=0, filename=filename, title=title, save=save,kmax=bandwidth,
                         plotme=plotme)
        return fcolors

    def plotHistogram(self, fcolors, nside3D, kmax, plotme=False):
        (mu, sigma) = norm.fit(fcolors)
        n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y)
        plt.xlim(mu - 4 * sigma, mu + 4 * sigma)
        plt.xlabel("Temperature/K")
        plt.ylabel("Frequency")
        plt.title(r"Histogram {}_{} HU Modeling of Planck SMICA Map ".format(kmax, nside3D), y=1.08)
        plt.savefig("./img1/Histogram_{}_{}.png".format(kmax, nside3D), dpi=300)
        if plotme:
            plt.show()

    def plot_aitoff(self, fcolors, kk, ll, mm, err,kmax, filename=None, title=None, plotme=False, save=False, nosigma=True):
        # noinspection PyUnresolvedReferences
        plt.clf()
        if title is None:
            title = "{}_{}_{}_{}".format(kmax, chg2ang(kk), chg2ang(ll), chg2ang(mm))

        f = plt.figure()
        if nosigma:
            hp.mollview(fcolors.squeeze(), title=title, min=-2 * self.sigma_smica,
                        max=2 * self.sigma_smica, unit="K", cmap=cm.RdBu_r)
        else:
            mu, sigma = norm.fit(fcolors.squeeze())
            hp.mollview(fcolors.squeeze(), title=title, min=mu - 2 * sigma,
                        max=mu + 2 * sigma, unit="K", cmap=cm.RdBu_r)
        hp.graticule()
        if save:
            plt.savefig(filename, format='png')
        if plotme:
            plt.show()
        f.clear()
        plt.close(f)

    def plot_Single_CL_From_Image(self, fcolors, nside, ymin=1E-6, ymax=1, xmax=300, log=False):      
        cl_SMICA, dl_SMICA, ell = self.get_dl(fcolors, nside)
        ymin=np.min(dl_SMICA[0:xmax])
        ymax=np.max(dl_SMICA[0:xmax])
        dl_SMICA = dl_SMICA / ymax
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        plt.plot(ell, dl_SMICA)
        ax = plt.gca()
        ax.set_ylabel("$\ell(\ell+1)C_\ell/2\pi \, \,(\mu K^2)$")
        ax.set_xlabel('$\ell$')
        ax.set_title("Angular Power Spectra")
        ax.legend(loc="upper right")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(left=1,right=xmax)
        ax.set_ylim(bottom=ymin,top=1)
        plt.show()
        return cl_SMICA, dl_SMICA, ell

    def plot_CL_From_Image(self, fcolors, nside, planck_theory_cl, xmax=300, xlim=2048, ymax=1.0, log=False):
        cl_SMICA, dl_SMICA, ell = self.get_dl(fcolors, nside)
        ymin=np.min(dl_SMICA[0:xmax])
        ymax=np.max(dl_SMICA[0:xmax])
        dl_SMICA = dl_SMICA / ymax
        planck_theory_cl[:, 1] = planck_theory_cl[:, 1] / np.max(planck_theory_cl[10:xlim, 1])

        x01 = np.array([2.14718116e-04, 9.58157432e-01])
        a=planck_theory_cl[:, 1]
        b= dl_SMICA[2:len(planck_theory_cl)+2]
        x00 = minimize(newerr, x01, args=( a, b),
                   method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        dl_SMICA1 = xx0[1]*dl_SMICA + xx0[0]
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.plot(ell, dl_SMICA1, color="green", label="Observation")
        ax.plot(planck_theory_cl[:, 0], planck_theory_cl[:, 1], color="red", label="Theory")
        ax.set_ylabel("$\ell(\ell+1)C_\ell/2\pi \, \,(\mu K^2)$")
        ax.set_xlabel('$\ell$')
        ax.set_title("Angular Power Spectra")
        ax.legend(loc="upper right")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(10, xlim)
        ax.set_ylim(1E-4, 1.2)
        ax.grid()
        plt.show()
        return cl_SMICA, dl_SMICA, ell, xx0

    def get_dl(self, fcolors, nside):
        cl_SMICA = hp.anafast(fcolors)
        ell = np.arange(len(cl_SMICA))
        pl = hp.sphtfunc.pixwin(nside=nside)
        # Deconvolve the beam and the pixel window function
        dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl ** 2)
        dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
        return cl_SMICA, dl_SMICA, ell

    def plot_ONLY_CL_From_Image(self, fcolors, nside, smica, nsidesmica, kmax, xmax=30):
        cl_SMICA, dl_SMICA_HU, ell = self.get_dl(fcolors.squeeze(), nside=nside)
        cl_SMICA, dl_SMICA, ell1 = self.get_dl(smica.squeeze(), nside=nsidesmica)
        dl_SMICA_HU /= np.max(dl_SMICA_HU[0:len(ell) // 2])
        dl_SMICA /= np.max(dl_SMICA[0:len(ell1) // 2])
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        line, = ax.plot(ell1, dl_SMICA)
        line.set_label('SMICA')
        ax1 = plt.twinx(ax)
        line1, = ax1.plot(ell, dl_SMICA_HU)
        line1.set_label('HU')
        ax1.set_ylabel('$\ell$')
        ax1.set_title("Angular Power Spectra {}_{}".format(self.kmax, self.nside3D))
        ax1.legend(loc="upper right")
        ax1.set_xlim(1, xmax)
        plt.ylim(1E-10, 1)
        ax1.grid()
        plt.savefig("./img1/AngularPowerSpectra_{}_{}.png".format(kmax, self.nside3D), dpi=300)
        plt.show()
        return dl_SMICA, dl_SMICA_HU, ell

    def plotSH(self, l, m, ll, lm, pp):
        fcolors = self.spherharmm(l, m, self.phi, pp)
        hp.mollview(fcolors.squeeze(), title="{}_{}_{}_{}".format(l, m, chg2ang(ll), chg2ang(lm)), cmap=cm.RdBu_r)
        hp.graticule()
        plt.show()
        return fcolors

    def plotNewMap(self, newmap, err, filename=None, title=None, plotme=False, save=False, nosigma=True):
        err0 = np.round(err, 3)
        
        if filename is None:
            filename = "./img1/aitoff_{}_{}_{}__{}_{}_{}.png".format(self.kmax, self.nside3D, err0,
                                                                     chg2ang(self.lambda_k),
                                                                     chg2ang(self.lambda_l),
                                                                     chg2ang(self.lambda_m))
        if title is None:
            title = "{}_{}_{}__{}_{}_{}".format(self.kmax, self.nside3D, err0, chg2ang(self.lambda_k),
                                                chg2ang(self.lambda_l),
                                                chg2ang(self.lambda_m))
        self.plot_aitoff(newmap, self.lambda_k, self.lambda_l,
                         self.lambda_m, self.kmax, err0, title=title, filename=filename, plotme=plotme, save=save, nosigma=nosigma)

    def calcStuff(self, alpha):
        err = 0.0
        results = 0.0
        prime = alpha[0]
        kmax = alpha[1]
        lk = prime[0]
        ll = prime[1]
        lm = prime[2]
        start_time = time()
        try:
            results, fcolors, err = self.project4D3d_0(kmax)
        except Exception as exception_object:
            print('Error with getting map for: {0}_{0}_{0}'.format(lk, ll, lm))
        stop_time = time()
        print(alpha, err)
        return [start_time, stop_time, err, prime, results]

    def mindict(self, test_dict):
        # Using min() + list comprehension + values()
        # Finding min value keys in dictionary
        temp = min(test_dict.values())
        res = [key for key in test_dict if test_dict[key] == temp]
        return res[0], test_dict[res[0]]

    def calcError(self, x, karray, nside3D):
        t1 = datetime.now()
        self.change_HSH_center(x[0], x[1], x[2], karray, nside3D, loadpriorG=False, doit=True, savePG=False)
        _, _, err = self.project4D3d_0(karray)
        print(x, err, (datetime.now() - t1).microseconds)
        return err

    def calcErrorDF(self, x):
        err = np.sum((np.dot(x.T, self.df[:, 4:]) - self.SMICA_LR.squeeze()) ** 2) * 1E6
        return err

    def cleanup(self, mypath, prefix):
        filelist = [join(mypath, f) for f in sorted(listdir(mypath)) if
                    isfile(join(mypath, f)) and f.startswith(prefix)]
        for f in filelist:
            if os.path.exists(f):
                os.remove(f)
            else:
                print("The file does not exist")

    # def get_gegenbauerAsyn(self):
    #     number_of_workers = 10
    #     multiindex = pd.MultiIndex.from_tuples(self.extra_G_keys(), names=('k', 'l'))
    #     columns = list(np.arange(len(self.cosksi)))
    #     if len(self.extra_G_keys()) == 0:
    #         return
    #     extraG = pd.DataFrame(data=np.zeros([len(multiindex), len(columns)]), index=multiindex, columns=columns)
    #     extra_G_keys = ((x,self.cosksi) for x in self.extra_G_keys())
    #     with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
    #         responses = executor.map(functionG, extra_G_keys)
    #     for response in responses:
    #         k = response[0][0]
    #         l = response[0][1]
    #         extraG.loc[(1+l,k-l),:] = response[1]
    #         print("transferred", k,l)
    #     return extraG

    def get_gegenbauerAsyn(self):
        multiindex = pd.MultiIndex.from_tuples(self.extra_G_keys(), names=('k', 'l'))
        columns = list(np.arange(len(self.cosksi)))
        if len(self.extra_G_keys()) == 0:
            return
        extraG = pd.DataFrame(data=np.zeros([len(multiindex), len(columns)]), index=multiindex, columns=columns)
        for key in self.extra_G_keys():
            a, extraG.loc[key, :] = functionG((key, self.cosksi))
        #             print(a, "transferred")
        return extraG

    def change_HSH_center(self, lk, ll, lm, karray, nside, doit=False, loadpriorG=False, savePG=False):
        kmax = max(karray)
        if not doit:
            array_number_match = len(self.karray) == len(karray)
            if array_number_match:
                array_match = np.sum(self.karray == karray) != 0
            else:
                array_match = False
            if (self.lambda_k, self.lambda_l, self.lambda_m, self.kmax) == (lk, ll, lm, kmax) \
                    and array_number_match and array_match:
                return
        self.loadpriorG = loadpriorG
        self.savePG = savePG
        if not loadpriorG:
            self.G = pd.DataFrame()
        self.nside3D = nside
        self.xx, self.yy, self.zz = hp.pix2vec(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
        self.theta3D, self.phi3D = hp.pix2ang(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
        self.lambda_k = lk
        self.lambda_l = ll
        self.lambda_m = lm
        self.z = self.zz + lk
        self.y = self.yy + ll
        self.x = self.xx + lm
        self.cosksi = np.cos(self.z)
        self.sinksi = np.sin(self.z)
        self.costheta = np.cos(self.y)
        self.phi = self.x
        G = {}
        listofk = sorted(karray)
        for k in listofk:
            for l in np.arange(1, k):
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1
        if len(G) != 0:
            self.extra_G_keys = G.keys
            if self.G.shape[0] == 0:
                self.G = self.get_gegenbauerAsyn()
            else:
                self.G = pd.concat([self.G, self.get_gegenbauerAsyn()], axis=0)
        if self.savePG:
            self.G.to_pickle(self.G_filename)
        # kjump = 20
        # ibeginning=0
        # for i in np.arange(kjump, self.kmax, kjump):
        #     newPfile = self.P_filename + "{}.npy".format(i)
        #     if not os.path.exists(newPfile):
        #         pp = get_legendre((0,np.arange(ibeginning,i), self.costheta))[1].squeeze()
        #         np.save(newPfile,pp)
        #         ibeginning=i

    # def get_P(self, l, costheta):
    #     pp = np.zeros([0, kmax + 1, kmax + 1])
    #     lencos = len(costheta)
    #     number_of_workers = 10
    #     mychunks = chunks(costheta, lencos // number_of_workers, l)
    #     P = {}
    #     with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
    #         responses = executor.map(get_legendre, mychunks)
    #     for response in responses:
    #         i = response[0]
    #         P[i] = response[1]
    #     for key in np.sort(list(P.keys())):
    #         pp = np.concatenate([pp, P[key]], axis=0)
    #     return pp

    def createGaussianBackground(self, x0, nside, delta, karray):
        xx, yy, zz = hp.pix2vec(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
        x, y, z = tuple(map(operator.add, (xx, yy, zz), x0))
        cosksi = np.cos(z)
        sinksi = np.sin(z)
        costheta = np.cos(y)
        phi = x
        ############################################################
        ############################################################
        # random amplitudes
        wavefunction = np.zeros(xx.shape)
        t1 = datetime.now()
        print((datetime.now() - t1).seconds)
        df = np.array([len(karray), len(xx)])
        self.kmax = np.max(karray)
        self.calc_hyperharmnano0(karray)

        ############################################################
        return wavefunction

    def project4D3d_1(self, karray):
        self.kmax = np.max(karray)
        self.calc_hyperharmnano1(karray)
        C = np.dot(self.df[:, 4:], self.SMICA_LR)
        B = np.dot(self.df[:, 4:], self.df[:, 4:].T)
        results = np.linalg.solve(B, C)

        self.newmap = np.dot(results.T, self.df[:, 4:])
        mu, sigma = norm.fit(self.newmap)
        self.newmap = (self.newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(self.newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        return results, self.newmap, err

    def project4D3d_0(self, karray):
        self.kmax = np.max(karray)
        self.calc_hyperharmnano2(karray)
        C = np.dot(self.df[:, 4:], self.SMICA_LR)
        B = np.dot(self.df[:, 4:], self.df[:, 4:].T)
        results = np.linalg.solve(B, C)

        self.newmap = np.dot(results.T, self.df[:, 4:])
        mu, sigma = norm.fit(self.newmap)
        self.newmap = (self.newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(self.newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        return results, self.newmap, err

    def calc_hyperharmnano0(self, karray):
        nnn = 20
        kmax = np.max(karray)
        npoints = len(self.xx)
        jj = 0
        for k in karray:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for l in llist:
                jj += 1
        self.df = np.zeros([jj, 4 + npoints])
        pp = np.array([legendre(kmax - 1, x, csphase=-1) for x in self.costheta])
        jj = 0
        for k in karray:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            jj = self.calc_hyperharmnano1(k, llist, pp, jj)

    def calc_hyperharmnano1(self, k, llist, pp, jj):
        ##############################################################
        ##############################################################
        newG = 0
        for ii, l in enumerate(llist):
            if (1 + l, k - l) not in self.G.index:
                aa, self.G[1 + l, k - l] = functionG(((1 + l, k - l), self.cosksi))
                newG += 1
            if (1 + l, k - l) not in self.G.index:
                print("missing", 1 + l, k - l)
            a = self.sinksi ** l * self.G.loc[(1 + l, k - l), :].values
            a1 = (-1) ** k * np.sqrt(2 * (k + 1) / np.pi * mp.factorial(k - l)
                                     * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
            a1 = float(a1)
            mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, len(llist)))))
            b = np.zeros(self.xx.shape)
            for m in mlist:
                b += a * a1 * self.spherharmm(l, m, self.phi, pp[:, l, np.abs(m)])
            c = b.std()
            mu = b.mean()
            if c != 0:
                self.df[jj, 0] = k
                self.df[jj, 1] = l
                self.df[jj, 2] = 0
                self.df[jj, 3] = c
                self.df[jj, 4::] = b / c
                jj += 1
            else:
                print("failed ", k, l)
        print(jj, self.df.shape)
        if newG != 0:
            self.G.to_pickle(self.G_filename)
        return jj

    def calc_hyperharmnano2(self, karray):
        npoints = len(self.xx)
        nnn = 20
        G = {}
        listofk = sorted(karray)
        lmax = np.max(karray)
        pp = np.array([legendre(lmax, x, csphase=-1) for x in self.costheta])
        ##############################################################
        gg = 0
        for k in listofk:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for ii, l in enumerate(llist):
                mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, nnn))))
                for m in mlist:
                    gg += 1
        self.df = np.zeros([gg, 4 + npoints])
        ##############################################################
        jj = 0
        for k in listofk:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for ii, l in enumerate(llist):
                if self.longG:
                    b = np.zeros(self.xx.shape)
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1.0
                    print(k, l)
                    continue
                if (1 + l, k - l) not in self.G.index:
                    print("missing", 1 + l, k - l)
                a = self.sinksi ** l * self.G.loc[(1 + l, k - l), :].values
                a1 = (-1) ** k * np.sqrt(2 * (k + 1) / np.pi * mp.factorial(k - l)
                                         * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
                a1 = float(a1)
                mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, nnn))))
                for m in mlist:
                    b = a * a1 * self.spherharmm(l, m, self.phi, pp[:, l, np.abs(m)])
                    c = b.std()
                    if c != 0:
                        self.df[jj, 0] = k
                        self.df[jj, 1] = l
                        self.df[jj, 2] = m
                        self.df[jj, 3] = c
                        self.df[jj, 4::] = b / c
                        jj += 1
                    else:
                        print("failed ", k, l, m)
        if len(G.keys()) != 0:
            self.extra_G_keys = G.keys
            print(G.keys())
            print("DO IT AGAIN")
        print(jj, gg, self.df.shape)

    def spherharmm(self, l, m, phi, pp):
        mm = np.abs(m)
        if m > 0:
            return pp * np.cos(mm * phi)
        if m == 0:
            return pp
        if m < 0:
            return pp * np.sin(mm * phi)

    def get_df_size(self, karray):
        npoints = len(self.xx)
        kmax = np.max(karray)
        jj = 0
        G = {}
        files = sorted([x for x in self.getSpectralFiles(self.mypath) if "/PG_data/P_{}_{}_{}_{}".format(self.nside3D,
                                                                                                         chg2ang(
                                                                                                             self.lambda_k),
                                                                                                         chg2ang(
                                                                                                             self.lambda_l),
                                                                                                         chg2ang(
                                                                                                             self.lambda_m)) in x])
        karray0 = [eval(x.replace(self.mypath, "").replace(".npy", "").split("_")[-1:][0]) for x in files]
        filesMap = {x: y for (x, y) in zip(karray0, files)}
        listofk = [x for x in sorted(list(filesMap.keys())) if x <= kmax]
        for k in listofk:
            pp = np.load(filesMap[k])
            llist = [np.count_nonzero(x, axis=0) for x in pp[0, :, :].squeeze()]
            pp = None
            llist = [x - 1 for x in llist if x < k + 1 and x > 1]
            for ii, l in enumerate(llist):
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1.0
                for m in np.arange(-l, l + 1):
                    if not self.longG:
                        jj += 1
                if self.longG:
                    jj += 1
        self.df = np.zeros([jj, 4 + npoints])
        self.extra_G_keys = G.keys

    def plot_orth(self, fcolors):
        sigma = np.std(fcolors)
        hp.orthview(fcolors, min=-2 * sigma, max=2 * sigma, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()

    def plot_aitoff_df(self, l, m, phi, pp=None, cmap=cm.RdBu_r):
        if pp is None:
            pp = np.array([legendre(l, x) for x in self.costheta])
        fcolors = self.spherharmm(l=l, m=m, phi=phi, pp=pp)
        sigma = np.std(fcolors)
        title = "{}_{}".format(l, m)
        hp.mollview(fcolors.squeeze(), title=title, min=-2 * sigma,
                    max=2 * sigma, unit="K", cmap=cm.RdBu_r)
        hp.graticule()
        self.plot_orth(fcolors)

    def optimizeNewMap(self, newmap0, SMICA_LR, xx0, nside3D, bandwidth, nosigma=True, mypath="./PG_data/"):
        newmap = xx0[1] * newmap0 + xx0[0]
        mu, sigma = norm.fit(newmap)
        newmap -= mu
        diffmap = SMICA_LR - newmap.squeeze()
        mu, sigma = norm.fit(diffmap)
        diffmap -= mu
        restoredmap = newmap + diffmap
        err = (1.0 - np.correlate(newmap.squeeze(), SMICA_LR.squeeze()) * 1e4)[0]
        return newmap, diffmap #, restoredmap, err

    def matchSMICA(self, a, newmap):
        diffmap_1024 = a * self.SMICA.squeeze() + newmap * (1 - a)
        diffmap_1024 = (diffmap_1024 - np.mean(diffmap_1024)) / np.std(diffmap_1024) * self.sigma_smica
        self.plotNewMap(diffmap_1024, err=0.0, plotme=True, title=str(a))

    def recalcXYZ(self, r):
        self.xx, self.yy, self.zz = hp.pix2vec(nside=self.nside3D, ipix=np.arange(hp.nside2npix(self.nside3D)))
        self.theta3D, self.phi3D = hp.pix2ang(nside=self.nside3D, ipix=np.arange(hp.nside2npix(self.nside3D)))
        self.z = r * self.zz + self.lambda_k
        self.y = r * self.yy + self.lambda_l
        self.x = r * self.xx + self.lambda_m
        self.cosksi = np.cos(self.z)
        self.sinksi = np.sin(self.z)
        self.costheta = np.cos(self.y)
        self.phi = self.x
        self.P = {}
        G = {}
        listofk = sorted(self.karray)
        for k in listofk:
            for l in np.arange(1, k):
                G[1 + l, k - l] = 1
        self.extra_G_keys = G.keys
        self.G = self.get_gegenbauerAsyn()

    def getCMBAtPos(self, lk, ll, lm, results):
        self.lambda_k = lk
        self.lambda_l = ll
        self.lambda_m = lm
        self.recalcXYZ(1.0)
        return self.getCMB(results).squeeze()


    def getCMB(self, results):
        self.calc_hyperharmnano2(self.karray)
        return self.df[:,4:].T.dot(results)


    def getCMBxyz(self, i, results):
        self.calc_hyperharmnano2(self.karray)
        self.newmap = np.dot(results.T, self.df[:, 4:])
        mu, sigma = norm.fit(self.newmap)
        self.newmap = (self.newmap - mu) / sigma * self.sigma_smica
        z = self.z - self.lambda_k
        y = self.y - self.lambda_l
        x = self.x - self.lambda_m
        df = np.zeros([len(x), 7])
        df[:, 0] = i
        df[:, 1] = x
        df[:, 2] = y
        df[:, 3] = z
        df[:, 4] = self.theta3D
        df[:, 5] = self.phi3D
        df[:, 6] = self.newmap
        return df, self.newmap
    
    
    def getUniverseMap(self, results, rr):
        map3D = np.zeros([0, 7])
        fcolors = {}
        for i in prange(len(rr)):
            self.recalcXYZ(rr[i])
            df, fcolors[i] = self.getCMBxyz(i, results)
            map3D = np.concatenate([map3D, df])
            print(i, rr[i], map3D.shape)
        return map3D, fcolors


class Color(Enum):
    FINDNEIGHBORHOOD = 1
    MINIMIZEPOSITION = 2
    EVALUATE_DF_AT_POSITION = 3
    CREATE_HIGH_RESOL_BACKGROUNDPLOT = 4
    CREATE_HISTOGRAM = 5
    CREATEMAPOFUNIVERSE = 6
    ################################
    FINDBESTFORKRANGE = 7
    OPTIMIZE_SPECTRUM = 8
    CREATE_GAUSSIAN_BACKGROUND = 9
    OPTIMIZE_SMICA_BACKGROUND = 10
    CREATE_VARIABLE_R_GIF = 11
    WORK_86_128 = 12
    CHARACTERIZE_SPECTRUM=13
    
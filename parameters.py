# Parameters
# Change the address where do you want to save your plots, data
import astropy.constants as cc
import astropy.units as uu
import numpy as np
import pandas as pd
import healpy as hp
from lib3 import B_l
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm

thishome = "./DataSupernovaLBLgov/"
planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + "COM_CMB_IQU-smica_1024_R2.02_full.fits", dtype=float)
(mu_smica, sigma_smica) = norm.fit(planck_IQU_SMICA)
planck_theory_cl = np.loadtxt(thishome +
            "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt", dtype=float)
imgAddress ='./img/'
supernovaAddress='./DataSupernovaLBLgov/'
sdssAddress='../sdss/'

time_after_decay=0

glueme=False
saveme = False  # Save my plot or data
loadme = False  # this switch is to be used if you want to read the saved excel file into a dataframe and go withou recalc
NumGalaxies = 10  # number of galaxies to be sampled in the 2-point correlation.  For professional calculation use 500
correctMe=True
d_units=1/uu.fm**3
d_Mev_units=1*uu.MeV/uu.fm**3

mp=(cc.m_p*cc.c**2).to("MeV")
mn=(cc.m_n*cc.c**2).to("MeV")
me=(cc.m_e*cc.c**2).to("MeV")
m_neutrino=0.086E-6*uu.MeV
m_hydrogen=cc.m_p+cc.m_e
K=236.0*uu.MeV
B=16.0*uu.MeV
L=50.0*uu.MeV
S=32.0*uu.MeV
MP= (cc.m_p*cc.c**2).to("MeV")
MN= (cc.m_n*cc.c**2).to("MeV")
ME= (cc.m_e*cc.c**2).to("MeV")
MNE = (cc.m_e*cc.c**2).to("MeV")
RR=14.03E9*uu.lyr.si
hydrogenatomwavelength = (cc.h/(m_hydrogen*cc.c)).si
deBroglieLambda= (cc.h/(m_hydrogen*cc.c)).si
n0= (1/deBroglieLambda**3).to(d_units) #  neutrons per uu.fm**3
# n0=0.16*d_units
# FORCING N0 TO BE THE PAPER'S VALUE
n0_MeV=n0*mn

T0=((3*np.pi**2*n0/2)**(2/3)*cc.hbar**2/(2*cc.m_n)).to("MeV")


# Blackholium stuff
# This is the Black Hole density where Fundamental Dilators are deBroglieLambda femtometer apart
# 8 x 1/8 + 6*1/2+1 of a FD per cell
# n0_y= (8/deBroglieLambda**3).to(d_units) #  flat hydrogen per uu.fm**3
dbh=5*m_hydrogen/deBroglieLambda**3
dbhMev_fm3=(dbh*cc.c**2).to(d_Mev_units)
dbh_y= (dbh/n0/m_hydrogen).si.value


# dneutron, dneutronMev_fm3, dilutionNeutron
# This is the Neutron Star density where Fundamental Dilators are 2*deBroglieLambda apart
# 8 * 1/8 a FD per cell
dneutron_y = dbh_y/2
dneutron=dneutron_y*n0*cc.m_n
dneutronMev_fm3=(dneutron*cc.c**2).to('MeV/fm**3')
dilutionNeutron=(dbh/dneutron)**(1/3)
neutronenergy=cc.m_n*cc.c**2

#  Number of Atoms per cubic meter at the Current Universe and density.

ccc=cc.c.si
GG=cc.G.si
pi=np.pi
lyr= uu.lyr.si

rho=(ccc**2/(0.776*2*pi**2*GG*RR**2)).si

oneATM_atoms=cc.N_A/0.0224*uu.mol/uu.m**3
numberOfAtoms=(rho/m_hydrogen).si
secondsInYear=365.25*24*3600
dilutionBlackholium=(rho/dbh)**(1/3)
dbh_radius=(dilutionBlackholium*RR).to('lyr')
dbh_t=(dbh_radius/cc.c).si


alpha = -2*(5*B*K - 3*(4*B - K)*T0)/(5*(9*B - K)*T0 + 3*T0**2)
eta = -18/5*(25*B**2 + 10*B*T0 + T0**2)/(5*(9*B - K)*T0 + 3*T0**2)
gamma = 1/9*(5*K + 6*T0)/(5*B + T0)
alpha_L = 1/18*(9*T0*alpha*(gamma - 1) + 2*T0*(3*gamma - 2) - 18*S*gamma + 6*L)/(T0*(gamma - 1))
eta_L = 1/18*(9*T0*eta*(gamma - 1) + 6*L - 18*S + 2*T0)/(T0*(gamma - 1))


glueme=False
saveme = False  # Save my plot or data
loadme = False  # this switch is to be used if you want to read the saved excel file into a dataframe and go withou recalc
NumGalaxies = 10  # number of galaxies to be sampled in the 2-point correlation.  For professional calculation use 500
correctMe=True

vssquared=[[58.86058360352017, 0.01072834645669285],
[74.16859657248726, 0.014862204724409411],
[164.9536822603057, 0.07342519685039367],
[236.94302918017598, 0.140255905511811],
[261.72996757758216, 0.1643700787401574],
[283.0176007410838, 0.19744094488188974],
[305.49791570171374, 0.2353346456692913],
[312.755905511811, 0.2945866141732283],
[318.7401574803149, 0.3249015748031496],
[324.65956461324686, 0.3359251968503937],
[481.1347846225103, 0.33730314960629915],
[797.6053728578045, 0.33730314960629915]
]
vssquarednp=np.array( [[ 1473.7549789124434 , 0.10357773147106887 ],
[ 1355.4456276610904 , 0.12191064237550966 ],
[ 1009.9848159721328 , 0.27097084132871874 ],
[ 881.3141184285442 , 0.374507550673963 ],
[ 848.6010539034853 , 0.4054257993026066 ],
[ 823.6413031407373 , 0.4443432736993886 ],
[ 799.8647019543275 , 0.48511302360304787 ],
[ 792.6814552176963 , 0.5427583386491895 ],
[ 786.9239849535443 , 0.5700013814045977 ],
[ 781.368969728491 , 0.5795905424093751 ],
[ 670.4124972623731 , 0.580778055375975 ],
[ 547.6160008534522 , 0.580778055375975 ]])

vssquaredpd=pd.DataFrame(vssquarednp,columns=["t","cs2"])



# alpha = 6.361862935613103
# eta = 4.715230561512334
# gamma = 1.1795526990793055
# eta_L = 1.7218097292135377
# alpha_L = 2.420566507439575
# T0 = 71.64729172270036*40* uu.MeV
# n0 = 0.43429495397* 1/uu.fm**3
# MN = 939.5654205203889 * uu.MeV
# MP = 938.2720881604905 * uu.MeV
# ME = 0.510998 * uu.MeV

# x0 = ((((mn - mp) / cc.hbar / cc.c / (3 * np.pi ** 2 * 0.5 * n0) ** (1 / 3))).si) ** 3
# d_units=1/uu.fm**3
# d_Mev_units=1*uu.MeV/uu.fm**3

# mp=(cc.m_p*cc.c**2).to("MeV")
# mn=(cc.m_n*cc.c**2).to("MeV")
# me=(cc.m_e*cc.c**2).to("MeV")
# m_neutrino=0.086E-6*uu.MeV
# m_hydrogen=cc.m_p+cc.m_e
# pi= np.pi


# deBroglieLambda= (cc.h/(m_hydrogen*cc.c)).si
# # n0= (1/deBroglieLambda**3/8).to(d_units) #  neutrons per uu.fm**3
# # FORCING N0 TO BE THE PAPER'S VALUE



# # Blackholium stuff
# # This is the Black Hole density where Fundamental Dilators are deBroglieLambda femtometer apart
# # 8 x 1/8 of a FD per cell
# # n0_y= (8/deBroglieLambda**3).to(d_units) #  flat hydrogen per uu.fm**3
# dbh=m_hydrogen/deBroglieLambda**3
# dbhMev_fm3=(dbh*cc.c**2).to(d_Mev_units)
# dbh_y=8

# cellvolume= deBroglieLambda**3
# numberofneutronspercell=1
# n0=(1/cellvolume/numberofneutronspercell**3).to(d_units)*8
# T0=((3*np.pi**2*n0/2)**(2/3)*cc.hbar**2/(2*cc.m_n)).to("MeV")

# # dneutron, dneutronMev_fm3, dilutionNeutron
# # This is the Neutron Star density where Fundamental Dilators are 2*deBroglieLambda apart
# # 8 * 1/8 a FD per cell
# dneutron=cc.m_n/cellvolume
# dneutronMev_fm3=(dneutron*cc.c**2).to('MeV/fm**3')
# dilutionNeutron=(dbh/dneutron)**(1/3)
# neutronenergy=cc.m_n*cc.c**2


# #  Number of Atoms per cubic meter at the Current Universe and density.

# ccc=cc.c.si
# GG=cc.G.si
# pi=np.pi
# lyr= uu.lyr.si
# RR=14.03E9*uu.lyr.si
# rho=(ccc**2/(0.776*2*pi**2*GG*RR**2)).si
# hydrogenatom=1.66E-24*uu.g
# oneATM_atoms=cc.N_A/0.0224*uu.mol/uu.m**3
# numberOfAtoms=(rho/hydrogenatom).si
# secondsInYear=365.25*24*3600
# # rho, numberOfAtoms

# tfactor=360

# dilutionBlackholium=(rho/dbh)**(1/3)
# dilutionBlackholium
# dbh_radius=(dilutionBlackholium*RR).to('lyr')
# dbh_t=(dbh_radius/cc.c).si
# ls=uu.lyr/secondsInYear
# BlackholiumRadiusinLightSeconds=dbh_radius.to(ls)
# NeutroniumRadius=BlackholiumRadiusinLightSeconds*dilutionNeutron
# NeutroniumTime= ((BlackholiumRadiusinLightSeconds*(dilutionNeutron-1))/cc.c).si
# print(n0, rho/cc.m_n)

import math

import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
import scipy.interpolate as sp
from scipy.optimize import minimize
from scipy.special import gamma as gammaF
from parameters import *


def volumeCalc(n, phi, r):
    if phi == 2:
        return 4 / 3 * np.pi * r ** 3
    return np.pi ** ((n - 1) / 2) * r ** 3 / gammaF((n - 1) / 2) * \
           integrate.quad(lambda x: np.sin(x) ** int(n - 2), 0, phi)[0]


def whatIsY(t):
    radius = (t * cc.c + dbh_radius).to("lyr")
    dilution = float((dbh_radius / radius) ** 3)
    y = dbh_y * dilution
    return y


from parameters import *

H0 = 1
c = 1
R0 = 1
pi4 = math.pi / 4.0
sqrt2 = math.sqrt(2)
RR = 14.03E9 * uu.lyr.si

today = 4.428e+17
today_y = whatIsY(today * uu.s)
today_y = today_y


def findjump(y):
    x0 = y[1]
    for x in y[2:]:
        if np.abs(x - x0) > 1:
            return [x0, x]
        x0 = x
    return []


class Universe():
    def __init__(self, eta, alpha, alpha_L, eta_L, T0, gamma, n0, vssquaredpd):
        self.k0 = 4 / 5
        #          Characterize Transparency.
        ionizationfraction = 0.5
        gammaT, z_transparency, TransparencyRadius, TransparencyTime, densityAtTransparency, \
        T_at_Transparency = findGammaT(ionizationfraction)
        ####################
        ####################
        ####################  TIMES AT TRANSPARENCY AND TODAY
        self.t_transparency = T_at_Transparency.value
        self.t_today = 2.725
        ####################
        ####################
        ####################
        self.gammaT = gammaT[0]
        self.TransparencyRadius = TransparencyRadius.value
        self.TransparencyTime = TransparencyTime.value
        self.z_transparency = z_transparency
        ####################  DENSITY AT TRANSPARENCY
        self.densityAtTransparency = densityAtTransparency.value
        ####################
        ####################
        ####################
        #       proton fraction calculation
        n = 1000
        protonfraction = np.linspace(1, 0, n)
        xout = findprotonfraction(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)

        self.densityBlackholium = dbh_y
        self.densityNeutronium = dneutron_y
        a=  xout.y.max() - xout.y.min()

        self.densityPreBigBang = xout.y.max()
        self.densityPostBigBang = xout.y.min()  # This is the optimized boundary for the Big Bang
        self.densityToday = today_y

        self.timePreBigBang = whatIsTime(self.densityPreBigBang)
        self.timePostBigBang = whatIsTime(self.densityPostBigBang)
        
        yy0 = np.linspace(self.densityBlackholium, self.densityPreBigBang, 100)
        yy = np.unique(np.concatenate([yy0, xout.y]))
        xin = pd.DataFrame(index=yy, columns=["y", "ProtonFraction"], dtype=float).fillna(0.0)
        xin.y = xin.index
        xin.loc[yy0[0:-1], "ProtonFraction"] = 0.0
        xin.loc[xout.y, "ProtonFraction"] = xout.ProtonFraction
        xin = xin.sort_values(by="y", ascending=False)

        ####################
        ####################V
        ####################  DENSITY AT FREEZING
        # Velocity of Sound
        x = xin.y.values
        y = self.vs(x, xin.ProtonFraction.values)

        ind = y > 1 / 3
        y[ind] = 1 / 3
        ind =(0.01 < y) * (y < 0.9999 / 3)
        self.densityAtPreFreezing = x[ind].max()
        self.densityAtFreezing = x[ind].min()

        # VS and ProtonFraction
        df = pd.DataFrame(columns=["t", "y", "r", "Vs", "ProtonFraction", "Energy", "Temperature", "Pressure"],
                          dtype=float)
        df.ProtonFraction = xin.ProtonFraction
        df.y = xin.y
        dff = pd.DataFrame(columns=["t", "y", "r", "Vs", "ProtonFraction", "Energy", "Temperature", "Pressure"],
                           dtype=float)
        dff.y = np.concatenate([np.geomspace(self.densityPostBigBang, self.densityAtTransparency, 300),
                                np.geomspace(densityAtTransparency, today_y, 300),
                                [self.densityBlackholium, self.densityNeutronium, self.densityAtPreFreezing,
                                 self.densityAtFreezing,
                                 self.densityPreBigBang, self.densityPostBigBang, self.densityAtTransparency,
                                 self.densityToday]])
        dff = dff.drop_duplicates(subset=["y"])
        dff.index = dff.y
        dff.ProtonFraction = 1.0
        dff.Vs = 0.0

        df = pd.concat([df, dff])
        df = df.drop_duplicates(subset=["y"])
        df.Vs = self.vs(df.y, df.ProtonFraction)
        ind = df.Vs >= 1 / 3
        df.loc[ind, "Vs"] = 1 / 3
        df.Energy = KE(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0) * df.y
        df.Pressure = Pressure(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
        df["Density"] = [(y * n0 * cc.m_n).to("kg/m**3").value for y in df.y]
        df["t"] = [whatIsTime(y) for y in df.y]
        df["r"] = [whatIsRadius(y) for y in df.y]
        df = df.sort_values(by="y", ascending=False)
        df = df.drop_duplicates(subset=["y"])
        self.df = df.copy()
        self.getEnergyPressure()

        # Derive Gamma from Energy and Pressure curves
        logP = np.array([np.log(x) for x in self.df.Pressure])
        logy = np.array([np.log(x) for x in self.df.y])
        dlogP = logP[1:] - logP[0:-1]
        dlogy = logy[1:] - logy[0:-1]
        self.df["gammaFromPressureY"] = None
        ll = dlogP / dlogy
        ll = np.append(ll, ll[-1])
        self.df.loc[:, "gammaFromPressureY"] = ll
        self.df = self.df.sort_values(by="y", ascending=False)
        self.df = self.df.reset_index(drop=True)
        self.df.index = self.df.y

        ################################
        ind = ["densityBlackholium", "densityNeutronium", "densityAtPreFreezing",
               "densityAtFreezing", "densityPreBigBang", "densityPostBigBang",
               "densityAtTransparency", "densityToday"]
        self.y_Seq = pd.DataFrame(data=np.zeros([8, 7]), index=ind,
                                  columns=["y", "Energy", "Pressure", "t", "radius", "Density", "Temperature"],
                                  dtype=float)

        self.y_Seq.y = [self.densityBlackholium, self.densityNeutronium, self.densityAtPreFreezing,
                        self.densityAtFreezing,
                        self.densityPreBigBang, self.densityPostBigBang, self.densityAtTransparency, self.densityToday]
        self.y_Seq = self.y_Seq.fillna(0.0)
        self.updateY_Seq()
        k0, t_today, t_transparency = self.find_k0()
        print(t_today, t_transparency)
        self.df["TemperatureDensity"] = self.df.Density * self.df.Temperature
        




    def vs(self, y, x):
        return 1/3*((15*(2*(eta-2*eta_L)*(x-1)*x-eta_L)*T0*(gamma-1)*gamma*y**(gamma-2)/n0+2*2**(2/3)*(x**(5/3)+(-x+1)**(5/3))*T0/(n0*y**(4/3)))*n0**2*y**2+6*(5*(2*(eta-2*eta_L)*(x-1)*x-eta_L)*T0*gamma*y**(gamma-1)/n0-2*2**(2/3)*(x**(5/3)+(-x+1)**(5/3))*T0/(n0*y**(1/3))-5*(2*(alpha-2*alpha_L)*(x-1)*x-alpha_L)*T0/n0)*n0**2*y)/((5*(2*(eta-2*eta_L)*(x-1)*x-eta_L)*T0*gamma*y**(gamma-1)-2*2**(2/3)*(x**(5/3)+(-x+1)**(5/3))*T0/y**(1/3)-5*(2*(alpha-2*alpha_L)*(x-1)*x-alpha_L)*T0)*n0*y-(3*2**(2/3)*(x**(5/3)+(-x+1)**(5/3))*T0*y**(2/3)+5*(2*(alpha-2*alpha_L)*(x-1)*x-alpha_L)*T0*y-5*(2*(eta-2*eta_L)*(x-1)*x-eta_L)*T0*y**gamma-5*MN*(x-1)+5*(ME+MP)*x)*n0)

    def pickleme(self):
        self.df.to_pickle("./df.pkl")
        self.y_Seq.to_pickle("./y_Seq.pkl")
        self.x_Seq.to_pickle("./x_Seq.pkl")

    def unpickleme(self):
        self.df = pd.read_pickle("./df.pkl")
        self.y_Seq = pd.read_pickle("./y_Seq.pkl")
        self.x_Seq = pd.read_pickle("./x_Seq.pkl")

    def createReport(self,  filename="./x_Seq.xls"):
        # Unit cell volume with deBroglieLambda side

        
        EnergyPerSupernova = 1E51*uu.erg
        energyPerNeutron = 0.78254809 * uu.MeV
        rho =  self.rho = self.y_Seq.loc["densityToday",  "Density"]*uu.kg/uu.m**3
        self. rho_atms = rho/cc.m_n
        VolumeHU = self.VolumeHU = self.y_Seq.loc["densityToday",  "HU Volume (cubic-light-years)"]*uu.lyr**3
        
        VolumeObservable = self.VolumeObservable = \
            self.y_Seq.loc["densityToday", "Observable Volume (cubic-light-years)"]*uu.lyr**3
            
        ratioHu = self.vol_ratio = (self.VolumeHU/self.VolumeObservable).si
        UniverseMassHU = self.UniverseMassHU = ( self.VolumeHU * self.rho).si
        UniverseMass = self.UniverseMass = (self.VolumeObservable * self.rho).si
        NumberOfNeutrons = self.numberOfNeutronsObs = (self.UniverseMass /cc.m_n).si
        NumberOfNeutronsHU = ratioHu * NumberOfNeutrons
        Energy = self.Energy = (self.numberOfNeutronsObs  * energyPerNeutron).si
        EnergyHU = self.EnergyHU = (self.Energy*self.vol_ratio).si
        
        NumberOfSupernovae =  self.numberOfSupernovae = (self.Energy /EnergyPerSupernova).si
        NumberOfSupernovaeHU = self.numberOfSupernovaeHU = self.vol_ratio * self.numberOfSupernovae
        
        
        BlackholiumRadius = self.y_Seq.loc["densityBlackholium", "radius"]*uu.lyr
        BigBangVolume = self.y_Seq.loc["densityPreBigBang", "Observable Volume (cubic-light-years)"]*uu.lyr**3
        BigBangVolumeHU = BigBangVolume* ratioHu
        SupernovaDensity =  self.supernovadensity = (self.numberOfSupernovae /BigBangVolume).to(1/uu.lyr**3)
        self.supernovaenergydensity = (self.Energy /BigBangVolume).to(uu.J/uu.lyr**3)

        cell = (deBroglieLambda) ** 3
        ls = uu.lyr/(365.25*24*3600)
#         A = ["rho", "rho_atms", "VolumeHU", "VolumeObservable", "UniverseMassHU", "UniverseMass", "Energy", "EnergyHU"]
#         B = [self.rho, self.rho_atms, self.VolumeHU, self.VolumeObservable, self.UniverseMassHU, self.UniverseMass,
#              self.Energy, self.EnergyHU]
#         for key, value in zip(A, B):
#             print(key, " = ", value.value, "=", value.unit)

        print("\n", 
                "Gamma Fitting" , "\n",
                "Plasma Gamma =", self.gamma0, "\n",
                "Hydrogen Gamma =", self.gamma1, "\n",
                "Adiabatic Boundary =", self.boundaryadiabatic.value, "=", self.boundaryadiabatic.si.unit, "\n",
                "Adiabatic Boundary_y =", self.boundaryadiabatic_y,  "\n",
                "Adiabatic Boundary_t =", self.boundaryadiabatic_t, "= seconds", "\n\n")

        print("\n", 
                "Initial Universe Properties", "\n",
                "Initial 4D Radius of the Universe (light-seconds) = ", (BlackholiumRadius).to(ls).value,"=ls", "\n",
                "Mass of the Observable Universe = ", UniverseMass, "=", UniverseMass.unit, "\n",
                "Mass of the Hyperspherical Universe = ", UniverseMassHU, "=", UniverseMassHU.unit,"\n\n",
             )        
        
        
        print("\n", 
                "General Properties", "\n",
                "EnergyPerSupernova = 1E51 =ergs",  "\n",
                "Cell Length (m) = ", deBroglieLambda.value,"=m", "\n", 
                "Current Density ($kg/m^3$) = ", rho.value, "=", rho.unit,  "\n",
                "Current Density ($1/m^3$) = ", self.rho_atms.si.value,"=1/m3", "\n\n" )
        
        print("\n", 
                "Current Universe Properties", "\n",
                "Current 4D Radius of the Universe (light-years) = ", 14.03E9 ,"=lyr", "\n", 
                "Current Density ($kg/m^3$) = ", rho.value, "=", rho.unit,  "\n",
                "Current Density ($1/m^3$) = ", self.rho_atms.si.value,"=1/m3", "\n\n" )

        
        print("\n", 
                "Observable Universe Properties", "\n",
                "Initial Volume Observable Universe = ", VolumeObservable.to(uu.lyr**3).value, "=lyr**3", "\n",
                "MassOfUniverse =", UniverseMass.to(uu.kg).value,"=kg", "\n",
                "Number of Neutrons =", NumberOfNeutrons, "\n",
                "BigBangEnergy = ", Energy.to(uu.J).value,"=J", "\n",
                "BigBangEnergyDensity = ",(EnergyPerSupernova*self.supernovadensity).to(uu.J/uu.lyr**3).value, "=J/lyr3", "\n",
                "Number of Supernovae = ", NumberOfSupernovae, "\n",
                "Big Bang Volume = ", BigBangVolume.to(uu.lyr**3).value ,"=", "lyr3", "\n",
                "Supernova Density (supernova per cubic lyr) =", SupernovaDensity.to(1/uu.lyr**3).value,"=1/lyr3", "\n\n" )
  
        
        print("\n", 
                "Hyperspherical Universe Properties", "\n",
                "Hyperspherical Universe", "\n",
                "Initial Volume HU = ", VolumeHU.to(uu.lyr**3).value,"=lyr3","\n",
                "MassOfUniverse HU =", UniverseMassHU.to(uu.kg).value,"=kg" "\n",
                "Number of Neutrons HU =", NumberOfNeutronsHU, "\n",
                "BigBangEnergy HU = ", EnergyHU.to(uu.J).value,"=J", "\n",
                "Big Bang Volume = ", BigBangVolumeHU.to(uu.lyr**3).value ,"=", "lyr3", "\n",
                "Number of Supernovae HU = ", NumberOfSupernovaeHU, "\n\n",)
        
        
        self.x_Seq.index = [x.replace("density", "").replace("At", "") for x in self.x_Seq.index]
        self.x_Seq.to_excel(filename)

    def getEnergyPressure(self):
        for i, row in self.df.iterrows():
            self.df.loc[self.df.y == row.y, "Energy"] = KE(row["y"], row["ProtonFraction"], eta, alpha, alpha_L, eta_L,
                                                           T0, gamma, n0)
            self.df.loc[self.df.y == row.y, "Pressure"] = Pressure(row["y"], row["ProtonFraction"], eta, alpha, alpha_L,
                                                                   eta_L, T0, gamma, n0).si

    #####################################################

    def getTemperature(self, x0=[1.28753349e+00, 1.333, 2.68717333e-08]):
        gcoef = x0[0]
        gammaT = x0[1]
        yboundary = x0[2]
        if yboundary > self.densityPreBigBang:
            yboundary= self.densityPreBigBang
        xprior = self.df.ProtonFraction.iloc[0]
        yprior = self.df.y.iloc[0]
        Temp_prior = self.df.loc[self.df.index[0], 'Temperature'] = 1E-4
        gamma_prior = self.df.gammaFromPressureY.iloc[0]
        Temp = Temp_prior
        sumofdx=0.0
        for i, row in list(self.df.iterrows())[1:]:
            y = row["y"]
            xnew = row["ProtonFraction"]
            if y >= self.densityPreBigBang:
                self.df.loc[y, "Temperature"] = Temp_prior
                yprior = y
                continue

            if y <= self.densityAtTransparency:
                gamma_prior = gammaT
            else:
                gamma_prior = gcoef

            if y < self.densityPreBigBang and y > yboundary:
                dx = row["ProtonFraction"] - xprior
                sumofdx +=dx
                dt =dx * ((MN - MP - ME) * 2 / 3 / cc.k_B).to(uu.K).value
                Temp = Temp_prior*(y/yprior)**(gamma_prior - 1) + dt
                xprior = row["ProtonFraction"]
                yprior = y


            if y < yboundary:
    #             Tmax =((MN - MP - ME) * 2 / 3 / cc.k_B).to(uu.K).value
                Temp = Temp_prior*(y/yprior)**(gamma_prior - 1)
            Temp_prior =Temp
            yprior = y
            self.df.loc[y,"Temperature"] = Temp
    #     print(Tmax, gamma_prior, sumofdx)
        t_today = self.df.loc[self.densityToday, "Temperature"]
        t_transparency = self.df.loc[self.densityAtTransparency, "Temperature"]
        return (self.t_today - t_today) ** 2 * 1E8 + (self.t_transparency - t_transparency) ** 2

    
    def updateY_Seq(self, cosmologicalangle=2):
        r = self.y_Seq.loc["densityToday", "Radius (lyr)"] = self.df.loc[self.densityToday, "r"]
        r_bigbang =  self.df.loc[self.densityPostBigBang, "r"]*uu.lyr
        volume_bigbang = 4/3 * np.pi * r_bigbang**3
        self.vol_ratio = ( volumeCalc(4, 2*np.pi, r)/(4/3*np.pi*r**3))
        energyPerNeutron = 0.78254809 * uu.MeV
        energyPerSupernova = 1E51 * uu.erg
        for name, yk in zip(self.y_Seq.index, self.y_Seq.y):
            E = self.y_Seq.loc[name, "Energy"] = self.df.loc[yk, "Energy"]
            P = self.y_Seq.loc[name, "Pressure"] = self.df.loc[yk, "Pressure"]
            t = self.y_Seq.loc[name, "t"] = self.df.loc[yk, "t"]
            y = self.y_Seq.loc[name, "y"] = self.df.loc[yk, "y"]
            r =self.y_Seq.loc[name, "radius"] = self.df.loc[yk, "r"]

            # print(r)
            #  Printable parameters
            self.y_Seq.loc[name, "Density"] = self.df.loc[yk, "Density"]
            self.y_Seq.loc[name, "n/n0"] = y
            self.y_Seq.loc[name, "$MeV/fm^3$"] = E
            self.y_Seq.loc[name, "$N/m^2$" ] = P
            self.y_Seq.loc[name, "Time (s)"] = t
            self.y_Seq.loc[name, "Radius (lyr)"] = self.df.loc[yk, "r"] = r
            self.y_Seq.loc[name, "Density ($Kg/m^3)$"] = (y*n0*cc.m_n).to (uu.kg/uu.m**3).value
            self.y_Seq.loc[name, "NeutronDensity ($1/m^3)$"] = (y * n0 ).to(1/ uu.m ** 3).value
            self.y_Seq.loc[name, "Temperature"] = self.y_Seq.loc[name, "Temperature K"] = self.df.loc[yk, "Temperature"]
            self.y_Seq.loc[name, "Density (1/fm3)"] =  self.df.loc[yk, "y"] * n0.to(d_units).value
            self.y_Seq.loc[name, "Time (year)"] = self.y_Seq.loc[name, "Time (s)"] / 365.25 / 24 / 3600
            self.y_Seq.loc[name, "Radius (light-seconds)"] = r * (365.25 * 24 * 3600)
            self.y_Seq.loc[name, "Observable Volume (cubic-light-years)"] = (4/3*np.pi*r**3)
            self.y_Seq.loc[name, "HU Volume (cubic-light-years)"] = volumeCalc(4, 2*np.pi, r)

        x_Seq_columns = ["n/n0", "$MeV/fm^3$", "$N/m^2$", "Time (s)", "Radius (lyr)", "Density ($Kg/m^3)$",
                          "Temperature K","Time (year)","Radius (light-seconds)",
                         "Observable Volume (cubic-light-years)",'HU Volume (cubic-light-years)', "NeutronDensity ($1/m^3)$"]
 
        
        self.x_Seq = self.y_Seq[x_Seq_columns]
        a=1


    def find_k0(self, x0 = [1.28753349e+00, 1.333, 2.68717333e-08]):
        results = scipy.optimize.minimize(self.getTemperature, x0, method="Nelder-Mead",
                                          options={'xatol': 1e-8, 'disp': True})
        self.k0 = results.x
        self.getTemperature(self.k0)
        self.updateY_Seq()
        t_today = self.df[self.df.y == self.densityToday].Temperature
        t_transparency = self.df[self.df.y == self.densityAtTransparency].Temperature
        self.gamma0 = self.k0[0]
        self.gamma1 = self.k0[1]
        self.boundaryadiabatic = (self.k0[2]*n0*cc.m_n).to(uu.kg/uu.m**3)
        self.boundaryadiabatic_y = self.k0[2]
        self.boundaryadiabatic_t = whatIsTime(self.k0[2])
        return self.k0, t_today, t_transparency


#     #####################################################


def KE(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    EKy = 3 / 5 * 2 ** (2 / 3) * (x ** (5 / 3) + (-x + 1) ** (5 / 3)) * T0 * y ** (2 / 3) + (
                2 * (alpha - 2 * alpha_L) * (x - 1) * x - alpha_L) * T0 * y - (
                      2 * (eta - 2 * eta_L) * (x - 1) * x - eta_L) * T0 * y ** gamma
    return EKy  # .to("MeV").value


def Pressure(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    P = -1 / 5 * (5 * (2 * (eta - 2 * eta_L) * (x - 1) * x - eta_L) * T0 * gamma * y ** (gamma - 1) / n0 - 2 * 2 ** (
                2 / 3) * (x ** (5 / 3) + (-x + 1) ** (5 / 3)) * T0 / (n0 * y ** (1 / 3)) - 5 * (
                              2 * (alpha - 2 * alpha_L) * (x - 1) * x - alpha_L) * T0 / n0) * n0 ** 2 * y ** 2
    return P  # .si


def dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # equilibrium equation is d(EK)/dx)= - y*n0*mu
    dEKy_x = 2 ** (2 / 3) * T0 * y ** (2 / 3) * (x ** (2 / 3) - (-x + 1) ** (2 / 3)) + 2 * (
                (alpha - 2 * alpha_L) * (x - 1) + (alpha - 2 * alpha_L) * x) * T0 * y - 2 * (
                         (eta - 2 * eta_L) * (x - 1) + (eta - 2 * eta_L) * x) * T0 * y ** gamma
    return dEKy_x


def findprotonfraction(xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # This calculated y values for protonfraction inputs - x is the protonfraction array [0,1]
    def mu(x, y):
        return (cc.hbar * cc.c * (3 * np.pi ** 2 * x * y * n0) ** (1 / 3)).to("MeV")

    def findy_err(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
        # equilibrium equation is d(EK)/dx + mu - (MN-MP)
        val = (dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu(x, y) - (MN - MP)).to("MeV").value
        return val

    df = {}
    y0 = 1.0

    for x in xx:
        try:
            # I am solving for y (density) and not x the protonfraction
            # root = scipy.optimize.root(findy, y0, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            root = scipy.optimize.brentq(findy_err, 0, 1, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            df[root] = x
        except Exception:
            pass
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns = ['ProtonFraction']
    df["y"] = df.index
    return df


# protonfraction = np.logspace(0,-3,300) #np.logspace(0,-7,1000) # np.geomspace(1, 1E-5, 1000)
# xout = findprotonfraction(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)


def findprotonfraction_y(yy, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # This calculated protonfraction values for densities y inputs - yy is the density array [0,8]
    def mu(x, y):
        return (cc.hbar * cc.c * (3 * np.pi ** 2 * (1 - x) * y * n0) ** (1 / 3)).to("MeV")

    def findy_err(x, y, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
        # equilibrium equation is d(EK)/dx + mu - (MN-MP)
        val = (dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu(x, y) - (MN - MP)).to("MeV").value
        return val

    df = {}
    for y in yy:
        try:
            # I am solving for y (density) and not x the protonfraction
            root = scipy.optimize.brentq(findy_err, 0, 1, args=(y, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            df[y] = root
            x0 = root
        except Exception:
            pass
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns = ['ProtonFraction']
    df["y"] = df.index
    return df


# xout = findprotonfraction_y(np.geomspace(1E-9,1e-2,1000), eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)


# Used for getting redshift associated with certain ionization fraction on the plasma (transparency epoch)
def findGammaT(ionizationfraction):
    # here x is the adiabatic cooling gamma and ionizationfraction is the ionizationfraction
    T_today = 2.72548  # today's temperature
    z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency = \
        findTransparencyEpoch(ionizationfraction)
    args = (ionizationfraction, z, T_today, TransparencyRadius, TransparencyTime, densityAtTransparency,
            T_at_Transparency),

    root = scipy.optimize.root(errfrac, x0=0.1, args=args)
    gammaT = root.x
    return gammaT, z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency


def errfrac(gammaIn, args):
    ionizationfraction, z, T_today, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency = args
    # https://www.schoolphysics.co.uk/age16-19/Thermal%20physics/Gas%20laws/text/Adiabatic_change_proof/index.html
    # TV**(gamma-1)=C_1
    # T_today*(RR**3)**(gamma-1)= T_at_transparency*(TransparencyRadius**3)**(gamma-1)
    # T_today = T_at_transparency*((TransparencyRadius/RR)**3)**(gamma-1)
    # since gamma = x
    # T_today = T_at_transparency*(TransparencyRadius/RR)**(3*gamma-3)
    errorgamma = T_today - ((TransparencyRadius / RR).si) ** (3 * gammaIn - 3) * T_at_Transparency.value
    return errorgamma


#############################################################
#############################################################  
def findTransparencyEpoch(ionizationfraction):
    x0 = 4000  # temperature
    root = scipy.optimize.root(fracIonization, x0=x0, args=(ionizationfraction))
    z = root.x[0]
    TransparencyRadius = (RR / (1 + z)).to("lyr")
    TransparencyTime = ((TransparencyRadius - dbh_radius) / cc.c).si
    densityAtTransparency = (densityUz(z) / n0).si
    T_at_Transparency = 2.72548 * uu.K * (1 + z)
    return z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency


#############################################################
#############################################################    
def densityUz(z):
    newradius = (RR / (1 + z)).to("lyr")
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbhMev_fm3 * dilution.si / mn).si
    return density


#############################################################
#############################################################
def fracIonization(z, ionizationfraction, fracH=0.92):
    # z is redshift and x is the ionization fraction of the plasma
    # start with the getting the density at the given z
    # from that one calculate the temperature for that z
    n = fracH* densityUz(z)
    T = 2.72548 * uu.K * (1 + z)
    kb = cc.k_B
    hbar = cc.hbar
    E = 13.6 * uu.eV
    # here we use the Saha equation
    A0 = ((me / cc.c ** 2 * 2 * np.pi * kb * T) / hbar ** 2) ** (3 / 2)  # equal 1/lambda**3
    A1 = (2 / n * A0 * np.exp(-(E / (kb * T)).si)).si
    # https://www.astro.umd.edu/~miller/teaching/astr422/lecture20.pdf equation 4
    # Saha equation https://en.wikipedia.org/wiki/Saha_ionization_equation
    # this is the number density
    A2 = ionizationfraction * ionizationfraction / (1 - ionizationfraction)
    return A2 - A1


#############################################################
#############################################################


def interpolateProtonFraction(y, lowestPF, densityPreBigBang, densityPostBigBang, f):
    if y >= densityPreBigBang:
        return lowestPF
    if y < densityPostBigBang:
        return 1.0
    return float(f(y))


def alphaZ(x):
    alpha = math.pi / 4 - math.asin(1 / math.sqrt(2) / (1 + x))
    return alpha


def z_Out_Of_Alpha(alpha):
    z = 1.0 / math.sin(pi4 - alpha) / sqrt2 - 1.0
    return z


def alpha_Out_Of_d_HU(d_HU):
    alpha = pi4 - np.asin((1.0 - d_HU) / sqrt2)
    return alpha


def z_Out_Of_d_HU(d_HU):
    alpha = alpha_Out_Of_d_HU(d_HU)
    z = z_Out_Of_Alpha(alpha)
    return z


def d_HU_epoch(R0, z):
    alpha = alphaZ(z)
    d_HU = R0 * (1 - math.cos(alpha) + math.sin(alpha))
    return d_HU


def actualVS(x, x0):
    beta = x0[0]
    n0 = x0[1]
    vs0 = 1 / np.sqrt(3)
    return vs0 / (1 + np.exp(beta * (x - n0)))


def findVSoundCurveParameters(yy):
    def errorf(x):
        error = 0
        for xx in yy.itertuples():
            error += (actualVS(xx.t, x) - xx.cs2) ** 2
        return error

    x0 = [6.48648947e-03, 1.05823880e+03]
    results = minimize(errorf, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

    df = {}
    for x0 in yy.itertuples():
        df[x0.t] = actualVS(x0.t, results.x)

    df1 = pd.DataFrame.from_dict(df, orient="index", columns=["cs2"])
    df1["t"] = df1.index

    return df1, results.x



def densityU(t, u):
    hydrogenatom = 1.66E-24 * uu.g
    newradius = t * cc.c * u + dbh_radius
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbhMev_fm3 * dilution.si)
    return density


def atmU(t, u):  # fraction of standard atmospherica pressure and number of atoms per cubic meter
    hydrogenatom = 1.66E-24 * uu.g
    newradius = t * cc.c * u + dbh_radius
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbh * dilution.si).si
    # fraction of standard atmospherica pressure
    numatm = (density / hydrogenatom / oneATM_atoms).si
    # number of atoms per cubic meter
    numatm_cubic_meter = (density / hydrogenatom).si
    return numatm, numatm_cubic_meter


def whatIsTemp(energy):
    kb = cc.k_B
    return (2 / 3 * energy / kb).si


def whatIsTime(y):
    dilution = float(y / dbh_y)
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    t = (radius - dbh_radius) / cc.c
    return t.si.value


def whatIsRadius(y):
    dilution = float(y / dbh_y)
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    return radius.to("lyr").value

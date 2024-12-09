from scipy.integrate import quad

from parameters import *


def PhaseVS(x1, x2, xSound):
    def f(s, xSound):
        t0 = dbh_t.value
        beta = xSound[0]
        n0 = xSound[1]
        vs0 = 1 / np.sqrt(3)
        A = 1 / (1 + np.exp(beta * (s - n0)))
        B = 1 / (s + t0)
        return vs0 * A * B
    return quad(f, x1, x2, args=(xSound))[0]


def windowdf(df, T,  a=100):
    b = int(2 // T) + a
    return df.iloc[a:b] - df.iloc[a]


def PhaseX(phase0, x, n, a1):
    a = 0
    x=x-a
    for j in range(1, n):
        i = j + 180
        a += np.cos(i * x) * np.sin(i * phase0) * filter(i, 4, 0.15)
    return a / n


def filter(i, n, b=0.0001):
    if n == 0:
        return i / (100 + i)
    if n == 1:
        return 1
    if n == 2:
        return np.exp(-b * (i - 100) ** 2)
    if n == 3:
        return np.sin(b * i) ** 2 / i ** 2
    if n == 4:
        return np.exp(-b * i)
    
    
def createSphere(a1,a2,a3,N,n,phase0):
    # sample spacing
    dec = np.linspace(-np.pi/2,np.pi/2, N)
    ra= np.linspace(-np.pi,np.pi,N)
    df = pd.DataFrame(columns=["dec", "ra"]) 
    a = np.meshgrid(dec,ra)
    df.dec = a[0].T.reshape(N*N)
    df.ra = a[1].T.reshape(N*N)
    df["z"]= np.sin(df.dec)
    a=np.sqrt(1-df.z*df.z)
    df["y"]= a* np.sin(df.ra)
    df["x"] = a * np.cos(df.ra)
    df["DensityZ"]= PhaseX(phase0,df.z, n, a1)
    df["radius"]= df.x**2+df.y**2+df.z**2
    df["DensityY"]= PhaseX(phase0,df.y, n, a2)
    df["DensityX"]= PhaseX(phase0,df.x, n, a3)
    df["density"]= df.DensityX + df.DensityY + df.DensityZ
    return df[["x","y","z", "density"]], df[["dec","ra","density"]], df
import numpy as np
import exoplanet as xo

G = 6.6743 * 10**(-8)        # cm^3 / (g * s^2)
rhosun = 1.4097798243075255  # g / cc
day = 86400.                 # seconds
rsun = 6.957 * 10**10        # cm
rearth = 6.378137 * 10**8    # cm

def simple_model(x, params, oversample=7, texp=None):
    ''' 
    Fit a light curve model across x values using exoplanet code by DFM
    Args:
        x: x values to evaluate light curve across
        params: dictionary containing info for constructing the light curve,
                i.e. RHOSTAR, PERIOD, T0, IMPACT, ROR, ECC, OMEGA, LD_U1, LD_U2, ROR
        synth: if you just want a clean synthetic curve to plot, regardless of exposure time;
               sets exposure time to a very small value (default = False)
        oversample: determines the assumed oversampling rate of the data (default = 7)
        texp: exposure time (in minutes); if None, then texp is measured from input data

    Out:
        light_curve: flux values of the light curve model
    '''   
    if texp is None:
        texp = np.min(np.diff(x))

    _keys = list(params.keys())
    if 'DUR14' in _keys:
        orbit = xo.orbits.SimpleTransitOrbit(period=params['PERIOD'], 
            t0=params['T0'], b=params['IMPACT'], ror=params['ROR'],
            duration=params['DUR14'])

    elif 'ECC' in _keys and 'OMEGA' in _keys:
        orbit = xo.orbits.KeplerianOrbit(rho_star=params['RHOSTAR'], period=params['PERIOD'], 
            t0=params['T0'], b=params['IMPACT'], ror=params['ROR'],
            ecc=params['ECC'], omega=params['OMEGA'])
    else:
        print('WARNING: dictionary of inputs requires either DUR14 or ECC & OMEGA!')
        return np.zeros(len(x))

    # Compute a limb-darkened light curve using starry
    light_curve = (
        xo.LimbDarkLightCurve(np.array([params['LD_U1'],params['LD_U2']]))
        .get_light_curve(orbit=orbit, r=params['ROR'], t=x, texp=texp, oversample=oversample)
        .eval()
    )

    return light_curve


def calc_aor(P, rho):
    '''
    Compute semimajor axis in units of R_star
    
    Args:
        P: period in units of days
        rho: stellar density in units of g/cc
    Out:
        aor: semimajor axis in units of R_star
    '''
    per = P * day
    aor = ( (per**2 * G * rho) / (3*np.pi) )**(1/3)
    return aor

def calc_T14(P, rho, b, ror, ecc, omega):
    '''
    T14 equation from Kipping 2014
    
    Args:
        P: period in units of days
        rho: stellar density in units of g/cc
        b: impact parameter
        ror: radius ratio
        ecc: eccentricity
        omega: argument of periastron in radians
    Out:
        T14: duration in units of days
    '''
    
    aor = calc_aor(P, rho)
    
    g_ew = (1 - ecc**2) / (1 + ecc * np.sin(omega))
    T14 = (P*day/np.pi) * g_ew**2 / (1 - ecc**2)**(1/2) * np.arcsin(( ( (1+ror)**2 - b**2 ) / ( aor**2*g_ew**2 - b**2 ) )**(1/2))
    
    return T14 / day

def calc_rho_star(P, T14, b, ror, ecc, omega):
    '''
    Inverting T14 equation from Kipping 2014
    
    Args:
        P: period in units of days
        T14: duration in units of days
        b: impact parameter
        ror: radius ratio
        ecc: eccentricity
        omega: argument of periastron in radians
    Out:
        rho_star: stellar density in units of g/cc
    '''
    
    oesw = 1 + ecc*np.sin(omega)
    oe2 = 1 - ecc**2
    oror = 1 + ror
    
    mult1 = ( oesw / oe2 )**3
    mult2 = 3*np.pi / ((P*day)**2 * G)
    mult3 = ( ( oror**2 - b**2 ) / np.sin( oesw**2 / oe2**(3/2) * T14*np.pi/P )**2 + b**2)**(3/2)
    
    return mult1 * mult2 * mult3


def calc_noise(T14, N, ror, SNR, texp=(30*60/day)):
    '''
    Estimate noise level from transit parameters and SNR
    Args:
        T14: duration in days
        N: number of transits
        ror: radius ratio
        SNR: signal to noise ratio
        texp: exposure time in days
    Out:
        noise: estimated sigma_noise
    '''
    
    noise = ( (T14/texp) * N )**(1/2) * ror**2 / SNR
    return noise

from   utils import *

import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm
import pymc3_ext as pmx

import argparse
import os



### Set up global arguments and paths ###

parser = argparse.ArgumentParser(description="simulated transit injection")
parser.add_argument("--sim_num", default=1, type=str, required=True)
parser.add_argument("--output_dir", default=None, type=str, required=True)

args = parser.parse_args()

SIM_NUM = int(args.sim_num)
PATH    = args.output_dir
NAME    = 'sim' + str(SIM_NUM)
os.makedirs(PATH, exist_ok=True)

print('path:', PATH)
print('name:', NAME)





### Get random transit parameters from a fixed kernel based on SIM_NUM ###

random = np.random.default_rng(SIM_NUM)

def impact_ks16(N=1):
    b = np.array(sorted(random.uniform(0, 0.99, N)))
    weights = (1-b**2)**(1/4)
    return random.choice(b, size=N, p=weights/sum(weights))

ror_inj   = np.exp(random.uniform(np.log(0.01), np.log(0.1))) # log-uniform distributed Rp/Rs [0.01 to 0.1]
per_inj   = np.exp(random.uniform(np.log(3), np.log(300)))    # log-uniform distributed periods [3 to 300 days]
ecc_inj   = random.beta(0.867, 3.03)                          # beta distributed eccentricity [Kipping 2013]
omega_inj = random.uniform(-0.5*np.pi, 1.5*np.pi)             # uniformly distributed omega [-pi/2 to 3pi/2]
imp_inj   = impact_ks16()[0]                                  # impact parameter distribution [Kipping & Sandford 2016]
snr_inj   = np.exp(random.uniform(np.log(10), np.log(100)))    # log-uniform distributed S/N ratio [10 to 100]

t0_inj    = 5.0    # arbitrary first transit midpoint [days]
rho_inj   = rhosun # sun-like stellar density [g/cc]
ld_u1_inj = 0.3    # arbitrary sun-like limb darkening coefficient 1
ld_u2_inj = 0.2    # arbitrary sun-like limb darkening coefficient 2
ld_u_vals = np.array([ld_u1_inj, ld_u2_inj]) # array of limb darkening coeffs

n_transit = 10      # number of transits to simulate
oversamp  = 11     # Kepler-like oversample rate
cadence   = 30.0   # Kepler-like 30-minute observation cadence
texp      = (60.*cadence)/day # exposure time in units of days





### Calculate additional parameters and compile info into a 'truths' dictionary ###

dur_inj = calc_T14(per_inj, rho_inj, imp_inj, ror_inj, ecc_inj, omega_inj) # calculated transit duration (1st-to-4th contact; days)
sig_inj = calc_noise(dur_inj, n_transit, ror_inj, snr_inj, texp)           # calculated photometric sigma_noise

truths = {
         'PERIOD':   per_inj,
         'T0':       t0_inj,
         'DUR14':    dur_inj,
         'ROR':      ror_inj,
         'IMPACT':   imp_inj,
         'ECC':      ecc_inj,
         'OMEGA':    omega_inj,
         'SNR':      snr_inj,
         'NOISE':    sig_inj,
         'RHOSTAR':  rho_inj,
         'LD_U1':    ld_u_vals[0],
         'LD_U2':    ld_u_vals[1]
         }

truths_df = pd.DataFrame([truths])
truths_df.index = [SIM_NUM]
truths_df.to_csv(os.path.join(PATH, NAME+'-truths.csv'))





### Generate synthetic lightcurve ###

n_samples = int((per_inj/texp) * (n_transit - 0.5))

x_test    = np.arange(n_samples)*texp
y_test    = sig_inj * random.normal(size=n_samples)+1
yerr_test = np.full(n_samples, sig_inj)

width = dur_inj * 3
x_fold = (x_test - t0_inj + 0.5*per_inj)%per_inj - 0.5*per_inj
m = np.abs(x_fold) < width

y_mod = simple_model(x_test, truths, oversample=oversamp, texp=texp).flatten()+1

data_df = pd.DataFrame()
data_df['X']      = x_test
data_df['Y']      = y_test*y_mod
data_df['Y_ERR']  = yerr_test

data_df['X_FOLD'] = x_fold
data_df['Y_INIT'] = y_test
data_df['Y_MOD']  = y_mod

data_df = data_df[m].copy()





### Plot and save synthetic lightcurve ###

data_df.sort_values('X_FOLD', inplace=True)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharex=True)

title = r'$simulation$ $\#{idx}$'.format(idx=SIM_NUM) + '\n'
title += r'P = {per} days; $R_{{\rm p}} =$ {rp} $R_\oplus$; S/N = {snr}'.format(
                                                               per=round(per_inj, 2),
                                                               rp=round(ror_inj*rsun/rearth, 1),
                                                               snr=round(snr_inj)) 
title += '\n' + r'b = {imp}; e = {ecc}; $\omega = {omega}\degree$'.format(
                                                               imp=round(imp_inj, 2),
                                                               ecc=round(ecc_inj, 2),
                                                               omega=round(omega_inj*180./np.pi))

axs[0].set_title(title, fontsize=18, pad=8)
axs[0].errorbar(data_df.X_FOLD, data_df.Y_INIT, yerr=data_df.Y_ERR, fmt='.', color='darkgrey', alpha=0.5, ms=10, label='simulated photometry')
axs[0].legend(fontsize=16)

axs[1].errorbar(data_df.X_FOLD, data_df.Y, yerr=data_df.Y_ERR, fmt='.', color='darkgrey', alpha=0.5, ms=10)
axs[1].plot(data_df.X_FOLD, data_df.Y_MOD, color='r', ls='-', lw=3, zorder=100, label='injected transit')
axs[1].legend(fontsize=16)
axs[1].set_xlabel("time since mid-transit [days]", fontsize=22)

for ax in axs:
    ax.set_xlim(-width, width)
    ax.set_ylabel("relative flux", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=16)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(PATH, NAME+'-lc.png'), dpi=200)
plt.close();

data_df.sort_values('X', inplace=True)
data_df.to_csv(os.path.join(PATH, NAME+'-lc_data.csv'))



print('Done!')


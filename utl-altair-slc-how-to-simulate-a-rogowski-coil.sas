%let pgm=utl-altair-slc-how-to-simulate-a-rogowski-coil;

%stop_submissiom;

Altair slc How to simulate a Rogowski Coil;

Hi res graphical output
https://github.com/rogerjdeangelis/utl-altair-slc-how-to-simulate-a-rogowski-coil/blob/main/rowgowski.png

Too long to post in a listserv, see github

github
https://github.com/rogerjdeangelis/utl-altair-slc-how-to-simulate-a-rogowski-coil

community.altair.com
https://community.altair.com/discussion/comment/195077?tab=all#Comment_195077?utm_source=community-search&utm_medium=organic-search&utm_term=slc

/*                   _
(_)_ __  _ __  _   _| |_
| | `_ \| `_ \| | | | __|
| | | | | |_) | |_| | |_
|_|_| |_| .__/ \__,_|\__|
        |_|
*/

/*--- these macro variables are passed to python ---*/

%let mu0 = 4 * np.pi * 1e-7;     /*--- Vacuum Permeability Constant ---*/
%let N = 200               ;     /*--- Number of turns              ---*/
%let r_mean = 0.08         ;     /*--- Mean radius (m)              ---*/
%let h_coil = 0.015        ;     /*--- Coil height (m)              ---*/
%let w_wire = 0.002        ;     /*--- Wire width (m)               ---*/

/*
 _ __  _ __ ___   ___ ___  ___ ___
| `_ \| `__/ _ \ / __/ _ \/ __/ __|
| |_) | | | (_) | (_|  __/\__ \__ \
| .__/|_|  \___/ \___\___||___/___/
|_|
*/

%utlfkil(d:/png/rowgowski.png);

%utl_slc_pybeginx(
   return=date             /*-  return date            -*/
  ,resolve=Y               /*- resolve macros in python-*/
  ,in=                     /*- input sas dataset       -*/
  ,out=rogowski            /*- output work.female      -*/
  ,py2r=c:/temp/py_dataframe.rds);/*- py 2 r dataframe -*/
cards4;
import pyreadstat as ps
import pandas as pd
import pyreadr as pr
from datetime import date

pyperclip.copy(date.today()) # clipboard to slc macro var
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Rogowski Coil Current Transformer parameters

mu0 = &mu0
N = &n                     # Number of turns
r_mean = &r_mean           # Mean radius (m)
h_coil = &h_coil           # Coil height (m)
w_wire = &w_wire           # Wire width (m)

A_cross = h_coil * w_wire  # Cross-sectional area (m**2)
M = mu0 * N * A_cross / (2 * np.pi * r_mean)  # Mutual inductance (H/m)1.2e-6 H/m

# Time setup for 60Hz power system simulation
fs = 20e3                  # 20kHz sampling
t = np.linspace(0, 0.1, int(0.1 * fs))  # 100ms capture
f_fund = 60                # Fundamental frequency

# Primary current: 100A distorted waveform (fundamental + harmonics)
i_primary = (100 * np.sin(2*np.pi*f_fund*t) +
             20 * np.sin(4*np.pi*f_fund*t) +
             10 * np.sin(6*np.pi*f_fund*t) +
             5 * np.sin(2*np.pi*1000*t))  # High frequency transient

# Rogowski voltage response: v(t) = -M * di/dt
di_dt = np.gradient(i_primary, t)
v_rogowski = -M * di_dt

# Add realistic noise and integrator non-idealities
noise = 0.5e-6 * np.random.normal(0, 1, len(t))  # 0.5uV noise floor
v_rogowski += noise

# Current reconstruction with analog integrator simulation (RC=1ms)
# Digital integrator with low-pass characteristics
alpha = 0.001  # Integrator time constant relative to sample period
i_reconstructed = np.zeros_like(t)
i_reconstructed[0] = 0
for n in range(1, len(t)):
    i_reconstructed[n] = (1-alpha) * i_reconstructed[n-1] + alpha * (v_rogowski[n] / M)

# DC offset removal (high-pass filter)
b_hp, a_hp = signal.butter(2, 1, btype='high', fs=fs)
i_reconstructed = signal.filtfilt(b_hp, a_hp, i_reconstructed)

# Performance metrics
rms_error = np.sqrt(np.mean((i_primary - i_reconstructed)**2))
rms_primary = np.sqrt(np.mean(i_primary**2))
accuracy_pct = (1 - rms_error/rms_primary) * 100

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Primary vs Reconstructed
axes[0,0].plot(t*1e3, i_primary, 'b-', linewidth=2, label='Primary (100A distorted)')
axes[0,0].plot(t*1e3, i_reconstructed, 'g--', linewidth=2, label=f'Reconstructed ({accuracy_pct:.1f}% accurate)')
axes[0,0].set_ylabel('Current (A)'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Rogowski voltage
axes[0,1].plot(t*1e3, v_rogowski*1e3, 'r-', linewidth=1.5, label='v_rogowski')
axes[0,1].set_ylabel('Voltage (mV)'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Error analysis
error = i_primary - i_reconstructed
axes[1,0].plot(t*1e3, error, 'm-', linewidth=1)
axes[1,0].set_xlabel('Time (ms)'); axes[1,0].set_ylabel('Error (A)')
axes[1,0].set_title(f'RMS Error: {rms_error:.2f}A ({rms_error/rms_primary*100:.2f}%)')
axes[1,0].grid(True, alpha=0.3)

# Frequency spectrum comparison
f = np.fft.rfftfreq(len(t), 1/fs)
P1 = np.abs(np.fft.rfft(i_primary))
P2 = np.abs(np.fft.rfft(i_reconstructed))
axes[1,1].semilogy(f/1e3, P1, 'b-', label='Primary', linewidth=2)
axes[1,1].semilogy(f/1e3, P2, 'g--', label='Reconstructed', linewidth=2)
axes[1,1].set_xlabel('Frequency (kHz)'); axes[1,1].set_ylabel('Magnitude')
axes[1,1].set_title('Frequency Response'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('d:/png/rowgowski.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()  # Prevents memory leaks in loops

print(f"Rogowski CT Accuracy: {accuracy_pct:.1f}%")
print(f"Mutual Inductance M: {M*1e6:.1f} uH/m")
print(f"Bandwidth: DC - {fs/2/1e3:.0f} kHz")
# Create a pandas DataFrame with all the data used in the plots
&out = pd.DataFrame({
    # Time data
    'time_s': t,
    'time_ms': t * 1e3,

    # Current data
    'i_primary': i_primary,
    'i_reconstructed': i_reconstructed,
    'error': i_primary - i_reconstructed,

    # Voltage data
    'v_rogowski': v_rogowski,
    'v_rogowski_mV': v_rogowski * 1e3,  # Convert to mV for plotting
    'di_dt': di_dt,

    # Noise component
    'noise': noise,

    # Parameters (repeated for each row for completeness)
    'M_H_per_m': np.full_like(t, M),
    'fs_Hz': np.full_like(t, fs),
    'f_fund_Hz': np.full_like(t, f_fund)
})
pr.write_rds("&py2r",&out) # panda datafame 2 r dataframe
;;;;
%utl_slc_pyendx

/*           _               _
  ___  _   _| |_ _ __  _   _| |_
 / _ \| | | | __| `_ \| | | | __|
| (_) | |_| | |_| |_) | |_| | |_
 \___/ \__,_|\__| .__/ \__,_|\__|
                |_|
*/
%put &=date;

Date=2025-12-14

Altair SLC

Rogowski CT Accuracy: -2124.0%
Mutual Inductance M: 0.0 uH/m
Bandwidth: DC - 10 kHz


YOU SHOULD BE ABLE TO USE THIS TABLE TO CREATE THE FOUR PLOTS
-------------------------------------------------------------

WORKX.ROGOWSKI    Observations    2000
TABLE             Variables       12

 -- NUMERIC --
                            Sample
Variable          Type       Value

TIME_S              N8    0.0499749875
TIME_MS             N8    49.974987494
I_PRIMARY           N8    -2.385500082
I_RECONSTRUCTED     N8    260.65594841
ERROR               N8    -263.0414485
V_ROGOWSKI          N8    -0.001418658
V_ROGOWSKI_MV       N8    -1.418658216
DI_DT               N8    94588.130749
NOISE               N8    1.6374552E-7
M_H_PER_M           N8          1.5E-8
FS_HZ               N8           20000
F_FUND_HZ           N8              60

/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/

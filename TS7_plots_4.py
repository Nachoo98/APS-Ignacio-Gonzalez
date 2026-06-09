import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io as sio
from scipy import signal

# ================================================================
# Parámetros
# ================================================================
fs    = 1000
gpass = 1
gstop = 40
ws1   = 0.1
wp1   = 0.5
ws2   = 45
wp2   = 35

wp = [wp1, wp2]
ws = [ws1, ws2]

# ================================================================
# Diseño filtros IIR (para ECG)
# ================================================================
sos_but    = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='butter',  output='sos', fs=fs)
sos_cauer  = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='cauer',   output='sos', fs=fs)
sos_cheby2 = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='cheby2',  output='sos', fs=fs)

# ================================================================
# Diseño filtros FIR
#   numtaps impar => Tipo I => demora entera
#   demora = (numtaps-1)//2 muestras
#   ECG: lfilter (causal) + índice compensado: ECG_f_fir[zoom + demora]
# ================================================================
numtaps = 3601
demora  = (numtaps - 1) // 2   # 1800 muestras

freq_bands = np.array([0.,  ws1, wp1, wp2, ws2, fs//2], dtype=float)
gains      = np.array([0,   0,   1,   1,   0,   0   ], dtype=float)

# Ventanas: boxcar (transición mínima, ripple alto), hamming, flattop (rizo plano, transición ancha)
b_boxcar  = signal.firwin2(numtaps, freq=freq_bands, gain=gains, nfreqs=2**14, window='boxcar',  fs=fs)
b_hamming = signal.firwin2(numtaps, freq=freq_bands, gain=gains, nfreqs=2**14, window='hamming', fs=fs)
b_flattop = signal.firwin2(numtaps, freq=freq_bands, gain=gains, nfreqs=2**14, window='flattop', fs=fs)

# FIR mínimos cuadrados: minimiza error L2 en las bandas especificadas
b_ls = signal.firls(numtaps, bands=freq_bands, desired=gains, fs=fs)

# ================================================================
# Vector de frecuencias con alta densidad en bandas de transición
# ================================================================
ww = np.concatenate([
    np.logspace(start=-2, stop=np.log10(wp1), num=500),
    np.linspace(start=wp1, stop=wp2, num=200),
    np.linspace(start=wp2, stop=fs/2, num=100),
])
ww = np.sort(np.unique(ww))

# Respuestas en frecuencia
_,     h_boxcar  = signal.freqz(b_boxcar,  worN=ww, fs=fs)
_,     h_hamming = signal.freqz(b_hamming, worN=ww, fs=fs)
w_fir, h_flattop = signal.freqz(b_flattop, worN=ww, fs=fs)
_,     h_ls      = signal.freqz(b_ls,      worN=ww, fs=fs)

phase_fir = np.unwrap(np.angle(h_flattop))

# ================================================================
# Plot 1: Módulo (veces) y Fase — flattop
# ================================================================
fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title(f"Respuesta en Frecuencia - FIR flattop (N={numtaps})")
ax1.plot(w_fir, abs(h_flattop), 'C0', label='Módulo')
ax1.set_ylabel("Módulo [Veces]", color='C0')
ax1.set_xlabel("Frecuencia [Hz]")
ax1.tick_params(axis='y', labelcolor='C0')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(w_fir, phase_fir, 'C1', label='Fase')
ax2.set_ylabel('Fase [rad]', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

# ================================================================
# Plot 2: dB + Fase desenvuelta — flattop
# ================================================================
fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title(f'Respuesta en Frecuencia Digital (dB) - FIR flattop (N={numtaps})')
ax1.plot(w_fir, 20 * np.log10(abs(h_flattop) + 1e-12), 'b')
ax1.set_ylabel('Amplitud [dB]', color='b')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(w_fir, phase_fir, 'g')
ax2.set_ylabel('Fase [rad]', color='g')
ax2.tick_params(axis='y', labelcolor='g')

nticks = 6
ax1.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.grid(True, linestyle='--', alpha=0.5)

# ================================================================
# Plot 3: Polos y Ceros + Retardo de Grupo
#   tf2zpk con N=3601 requiere encontrar 3600 raíces => muy lento
#   => se diseña un filtro auxiliar con N=301 solo para el diagrama
# ================================================================
fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)

ax_z = fig_sys.add_subplot(1, 2, 1)
numtaps_pz = 1001
b_pz = signal.firwin2(numtaps_pz, freq=freq_bands, gain=gains, nfreqs=2**14, window='boxcar', fs=fs)
ceros, polos, k = signal.tf2zpk(b_pz, [1])

circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax_z.add_artist(circulo)
ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=5, fillstyle='none', label='Ceros')
ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=5, mew=2,            label='Polos')
ax_z.axhline(0, color='black', linewidth=0.5)
ax_z.axvline(0, color='black', linewidth=0.5)
ax_z.set_title(f'Diagrama Polos/Ceros - FIR boxcar (N={numtaps_pz})')
ax_z.set_xlabel('Parte Real')
ax_z.set_ylabel('Parte Imaginaria')
ax_z.set_aspect('equal', adjustable='box')
ax_z.set_xlim([-1.2, 1.2])
ax_z.set_ylim([-1.2, 1.2])
ax_z.grid(True, alpha=0.5)
ax_z.legend()

ax_gd = fig_sys.add_subplot(1, 2, 2)
gd_fir = -np.diff(phase_fir) / np.diff(2 * np.pi * ww / fs)
gd_fir = np.append(gd_fir[0], gd_fir)
ax_gd.plot(ww, gd_fir, 'm', linewidth=2, label='Group delay')
ax_gd.axhline(demora, color='k', linestyle='--', alpha=0.7, label=f'demora = {demora} muestras')
ax_gd.set_title(f'Retardo de Grupo - FIR flattop (N={numtaps})')
ax_gd.set_xlabel('Frecuencia [Hz]')
ax_gd.set_ylabel('Retardo [Muestras]')
ax_gd.set_xlim(0, fs/2)
ax_gd.set_ylim(0, 1.1 * np.max(gd_fir))
ax_gd.grid(True, alpha=0.5)
ax_gd.legend()

# ================================================================
# Plot 4: Plantilla + comparación de ventanas + FIR LS (escala log en X)
# ================================================================
fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title(f"Comparación FIR (N={numtaps}): ventanas y mínimos cuadrados")

piso_grafico  = -125
techo_grafico = 10
x_min_log     = 0.05   # mínimo del eje log (no puede ser 0)

ax1.plot(w_fir, 20 * np.log10(np.abs(h_boxcar)  + 1e-12), 'C2', linewidth=1.2, label='boxcar',  alpha=0.85)
ax1.plot(w_fir, 20 * np.log10(np.abs(h_hamming) + 1e-12), 'C3', linewidth=1.2, label='hamming', alpha=0.85)
ax1.plot(w_fir, 20 * np.log10(np.abs(h_flattop) + 1e-12), 'C0', linewidth=1.5, label='flattop')
ax1.plot(w_fir, 20 * np.log10(np.abs(h_ls)      + 1e-12), 'C1', linewidth=1.5, label='FIR LS', linestyle='--')

ax1.fill_between([x_min_log, ws1], -gstop,        techo_grafico,  color='red',   alpha=0.20)
ax1.plot(        [x_min_log, ws1], [-gstop,       -gstop],        'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1,  wp2],      piso_grafico,  -gpass,         color='red',   alpha=0.20)
ax1.plot(        [wp1,  wp2],      [-gpass,       -gpass],         'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1,  wp2],      -gpass,        techo_grafico,  color='green', alpha=0.15)
ax1.fill_between([ws2,  fs/2],     -gstop,        techo_grafico,  color='red',   alpha=0.20)
ax1.plot(        [ws2,  fs/2],     [-gstop,       -gstop],        'k--', linewidth=1, alpha=0.7)

ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_xscale('log')
ax1.set_ylabel('Amplitude in dB')
ax1.set_xlabel('Frequency [Hz]  [escala log]')
ax1.set_xlim(x_min_log, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico])
ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

# ----------------------------------------------------------------
# Plot 4b: misma plantilla pero escala lineal — muestra rizos del stopband
# ----------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title(f"Comparación FIR (N={numtaps}) — escala lineal (rizos en stopband)")

ax1.plot(w_fir, 20 * np.log10(np.abs(h_boxcar)  + 1e-12), 'C2', linewidth=1.2, label='boxcar',  alpha=0.85)
ax1.plot(w_fir, 20 * np.log10(np.abs(h_hamming) + 1e-12), 'C3', linewidth=1.2, label='hamming', alpha=0.85)
ax1.plot(w_fir, 20 * np.log10(np.abs(h_flattop) + 1e-12), 'C0', linewidth=1.5, label='flattop')
ax1.plot(w_fir, 20 * np.log10(np.abs(h_ls)      + 1e-12), 'C1', linewidth=1.5, label='FIR LS', linestyle='--')

ax1.fill_between([0,    ws1],  -gstop,       techo_grafico,  color='red',   alpha=0.20)
ax1.plot(        [0,    ws1],  [-gstop,      -gstop],        'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1,  wp2],  piso_grafico, -gpass,         color='red',   alpha=0.20)
ax1.plot(        [wp1,  wp2],  [-gpass,      -gpass],         'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1,  wp2],  -gpass,       techo_grafico,  color='green', alpha=0.15)
ax1.fill_between([ws2,  fs/2], -gstop,       techo_grafico,  color='red',   alpha=0.20)
ax1.plot(        [ws2,  fs/2], [-gstop,      -gstop],        'k--', linewidth=1, alpha=0.7)

ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_ylabel('Amplitude in dB')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_xlim(0, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico])
ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

# ================================================================
# Carga y filtrado del ECG
#   FIR: lfilter (causal, una pasada) => retardo = demora muestras
#   Para alinear, graficar ECG_f_fir[zoom_region + demora]
# ================================================================
fs_ecg = 1000
mat_struct    = sio.loadmat(r'C:\Users\MSI\Desktop\ECG_TP4.mat')
ecg_one_lead  = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

ECG_f_butt   = signal.sosfiltfilt(sos_but,    ecg_one_lead)
ECG_f_cauer  = signal.sosfiltfilt(sos_cauer,  ecg_one_lead)
ECG_f_cheby2 = signal.sosfiltfilt(sos_cheby2, ecg_one_lead)
ECG_f_fir    = signal.lfilter(b_flattop, [1], ecg_one_lead)

# ================================================================
# Regiones sin ruido — FIR con compensación de demora
# ================================================================
regs_sin_ruido = (
    [4000, 5500],
    [10_000, 11_000],
)

for ii in regs_sin_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    fir_idx     = np.clip(zoom_region + demora, 0, cant_muestras - 1)

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG',         linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region],   label='Butterworth')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region],  label='Cauer')
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    plt.plot(zoom_region, ECG_f_fir[fir_idx],        label='FIR (flattop)')
    plt.title(f'ECG sin ruido — muestras {ii[0]} a {ii[1]}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend()
    plt.gca().set_yticks(())

# ================================================================
# Regiones con ruido — FIR con compensación de demora
# ================================================================
regs_con_ruido = (
    np.array([5,   5.2]) * 60 * fs_ecg,
    np.array([12, 12.4]) * 60 * fs_ecg,
    np.array([15, 15.2]) * 60 * fs_ecg,
)

for ii in regs_con_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    fir_idx     = np.clip(zoom_region + demora, 0, cant_muestras - 1)

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG',         linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region],   label='Butterworth')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region],  label='Cauer')
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    plt.plot(zoom_region, ECG_f_fir[fir_idx],        label='FIR (flattop)')
    plt.title(f'ECG con ruido — muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend()
    plt.gca().set_yticks(())

plt.show()

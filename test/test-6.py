import matplotlib
matplotlib.use('TkAgg')  # ventanas emergentes (cambiar a 'Qt5Agg' si no funciona)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io as sio
from scipy import signal

# ================================================================
# Diseño del filtro (fs = 1000 Hz, igual que ECG)
# ================================================================
fs    = 1000
gpass = 1
gstop = 40
ws1   = .1
wp1   = .5
ws2   = 45
wp2   = 35

wp = [wp1, wp2]
ws = [ws1, ws2]

# gpass/2 y gstop/2: sosfiltfilt aplica el filtro dos veces, duplicando la atenuacion
sos_but    = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='butter',  output='sos', fs=fs)
sos_cauer  = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='cauer',   output='sos', fs=fs)
sos_cheby2 = signal.iirdesign(wp, ws, gpass/2, gstop/2, analog=False, ftype='cheby2',  output='sos', fs=fs)

# Diseno de FIR


# PREGUNTAS
# Con =30 que pasa? con 31? que tipo de FIR es? con 30 es II) y con 31 es I)
# en cantidad de coef pares, la demora del filtro es entera o no entera? No entero porque es (numtaps-1)/2
# Un filtro II tiene un cero topologico en Ny si o no?  Si, tiene un cero topologico en Ny.

numtaps=30

gains=10**((-1)*np.array([gstop, gstop, gpass,gpass, gstop, gstop])/20)

#gains_2 = np.array([0, 0, 1,1, 0, 0])

if numtaps % 2 ==0:
    gains[-1]=0
b_win= signal.firwin2(numtaps, freq = np.array([0., ws1, wp1, wp1, ws2, fs//2]), gain=gains, fs=fs)

sos = sos_but  # filtro activo para los graficos de respuesta

# Vector de frecuencias con alta densidad en bandas de transicion
ww = np.concatenate([
    np.logspace(start=-2, stop=np.log10(wp1), num=500),
    np.linspace(start=wp1, stop=wp2, num=200),
    np.linspace(start=wp2, stop=fs/2, num=100),
])
ww = np.sort(np.unique(ww))

w, h  = signal.sosfreqz(sos, worN=ww, fs=fs)
phase = np.unwrap(np.angle(h))

# ================================================================
# Respuesta en frecuencia: Modulo (veces) y Fase
# ================================================================
fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title("Respuesta en Frecuencia del Filtro IIR")
ax1.plot(w, abs(h), 'C0', label='Módulo')
ax1.set_ylabel("Módulo [Veces]", color='C0')
ax1.set_xlabel("Frecuencia [Hz]")
ax1.tick_params(axis='y', labelcolor='C0')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(w, phase, 'C1', label='Fase')
ax2.set_ylabel('Fase [rad]', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

# ================================================================
# Respuesta en frecuencia: Escala dB + Fase desenvuelta
# ================================================================
fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title('Respuesta en Frecuencia Digital (Escala dB)')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitud [dB]', color='b')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(w, phase, 'g')
ax2.set_ylabel('Fase [rad]', color='g')
ax2.tick_params(axis='y', labelcolor='g')

nticks = 6
ax1.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.yaxis.set_major_locator(ticker.LinearLocator(nticks))
ax2.grid(True, linestyle='--', alpha=0.5)

# ================================================================
# Diagrama de Polos y Ceros (Plano z) + Retardo de Grupo
# ================================================================
fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)

ax_z = fig_sys.add_subplot(1, 2, 1)
ceros, polos, k = signal.sos2zpk(sos)

circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax_z.add_artist(circulo)
ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=8, fillstyle='none', label='Ceros')
ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, mew=2,            label='Polos')
ax_z.axhline(0, color='black', linewidth=0.5)
ax_z.axvline(0, color='black', linewidth=0.5)
ax_z.set_title('Diagrama de Polos y Ceros (Plano z)')
ax_z.set_xlabel('Parte Real')
ax_z.set_ylabel('Parte Imaginaria')
ax_z.axis('equal')
ax_z.set_xlim([-1.2, 1.2])
ax_z.set_ylim([-1.2, 1.2])
ax_z.grid(True, alpha=0.5)
ax_z.legend()

ax_gd = fig_sys.add_subplot(1, 2, 2)
gd = -np.diff(phase) / np.diff(ww)
gd = np.append(gd[0], gd)
ax_gd.plot(ww, gd, 'm', linewidth=2)
ax_gd.set_title('Retardo de Grupo (Group Delay)')
ax_gd.set_xlabel('Frecuencia [Hz]')
ax_gd.set_ylabel('Retardo [Muestras]')
ax_gd.set_xlim(0, fs/2)
ax_gd.grid(True, alpha=0.5)

# ================================================================
# Respuesta con plantilla de diseño sombrada
# ================================================================
fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title("Frequency Response of IIR Filter (SOS Structure)")
ax1.plot(w, 20 * np.log10(np.abs(h)), 'b', linewidth=1.8, label='Filtro diseñado')

piso_grafico  = -125
techo_grafico = 10

ax1.fill_between([0,   ws1], piso_grafico, -gstop,        color='red',   alpha=0.15)
ax1.plot(        [0,   ws1], [-gstop,      -gstop],        'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1, wp2], piso_grafico, -gpass,         color='red',   alpha=0.15)
ax1.plot(        [wp1, wp2], [-gpass,      -gpass],         'k--', linewidth=1, alpha=0.7)
ax1.fill_between([wp1, wp2], 3,             techo_grafico,  color='green', alpha=0.15)
ax1.plot(        [wp1, wp2], [3,            3],             'k--', linewidth=1, alpha=0.7)
ax1.fill_between([ws2, fs/2], piso_grafico, -gstop,        color='red',   alpha=0.15)
ax1.plot(        [ws2, fs/2], [-gstop,      -gstop],        'k--', linewidth=1, alpha=0.7)

ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_ylabel('Amplitude in dB', color='b')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_xlim(0, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico])
ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

# ================================================================
# Carga y filtrado del ECG
# ================================================================
fs_ecg = 1000
mat_struct    = sio.loadmat(r'C:\Users\MSI\Desktop\ECG_TP4.mat')
ecg_one_lead  = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

ECG_f_butt   = signal.sosfiltfilt(sos_but,    ecg_one_lead)
ECG_f_cauer  = signal.sosfiltfilt(sos_cauer,  ecg_one_lead)
ECG_f_cheby2 = signal.sosfiltfilt(sos_cheby2, ecg_one_lead)

# ================================================================
# Regiones sin ruido (por muestra)
# ================================================================
regs_sin_ruido = (
    [4000, 5500],
    [10_000, 11_000],
)

for ii in regs_sin_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG',         linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region],   label='Butterworth')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region],  label='Cauer')
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    plt.title(f'ECG sin ruido — muestras {ii[0]} a {ii[1]}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend()
    plt.gca().set_yticks(())

# ================================================================
# Regiones con ruido (por minuto)
# ================================================================
regs_con_ruido = (
    np.array([5,   5.2]) * 60 * fs_ecg,
    np.array([12, 12.4]) * 60 * fs_ecg,
    np.array([15, 15.2]) * 60 * fs_ecg,
)

for ii in regs_con_ruido:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG',         linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region],   label='Butterworth')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region],  label='Cauer')
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    plt.title(f'ECG con ruido — muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend()
    plt.gca().set_yticks(())

plt.show()  # muestra todas las ventanas juntas al final

# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:53:32 2026

@author: JGL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as spsw
 
fs = 1000   # frecuencia de muestreo [Hz]
N  = 100    # cantidad de muestras
n  = np.arange(N)
 
# La señal: coseno a 205 Hz
# 205 Hz NO cae en un bin exacto (bin 20 = 200 Hz, bin 21 = 210 Hz)
x = np.cos(2 * np.pi * 205 * n / fs)
 
freqs = np.fft.rfftfreq(N, d=1/fs)  # eje de frecuencias: 0 a fs/2
 
# -----------------------------------------------------------
# GRÁFICO 1: qué pasa con cada ventana
# Todo el bloque de plot tiene que estar DENTRO del for
# -----------------------------------------------------------
for nombre, w in [("Rectangular", np.ones(N)),
                  ("Hann",        spsw.hann(N, sym=False))]:
 
    X = np.fft.rfft(x * w) / N          # FFT con la ventana aplicada
 
    plt.figure(figsize=(8, 3))
    plt.plot(freqs, 20 * np.log10(np.abs(X) + 1e-12), label=nombre)
    plt.axvline(205, color="red", linestyle="--", label="205 Hz real")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)| [dB]")
    plt.title(f"Espectro con ventana {nombre}")
    plt.ylim(-80, 0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
 
plt.show()
 
# -----------------------------------------------------------
# ¿QUÉ TENÉS QUE MIRAR?
#
# La línea roja vertical es donde DEBERÍA estar el pico (205 Hz).
#
# Ventana RECTANGULAR:
#   - El pico no está exactamente en 205 Hz sino entre dos bins
#     (200 y 210 Hz). Está "aplastado".
#   - Aparecen "patas" que se extienden por todo el espectro
#     (decaen lento, tipo -6 dB por octava).
#   - Eso es el DESPARRAMO: energía que aparece en frecuencias
#     donde no hay señal real.
#
# Ventana HANN:
#   - El pico también está entre los dos bins, pero es MÁS ANCHO.
#   - Las "patas" casi desaparecen (caen a -80 dB rápido).
#   - La ventana suaviza el desparramo a costa de ensanchar el pico.
#
# CONCLUSIÓN VISUAL:
#   Rectangular → pico estrecho pero patas grandes
#   Hann        → pico ancho pero patas pequeñas
#   Ese es el trade-off fundamental de las ventanas.
# -----------------------------------------------------------
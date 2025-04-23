# EXP.NO.8-Simulation-of-QPSK

8.Simulation of QPSK

# AIM

To simulate and visualize the Quadrature Phase Shift Keying (QPSK) signal using Python.

# SOFTWARE REQUIRED

Python (version 3.6 or above)

NumPy – for numerical operations and signal processing

# ALGORITHMS

Generate Random Bit Sequence:

Generate a sequence of random binary bits.

Each QPSK symbol requires 2 bits.

Map Bits to QPSK Symbols:

Combine every two bits into one QPSK symbol.

Use the mapping:

00 → 0 radians

01 → π/2 radians

10 → π radians

11 → 3π/2 radians

Generate QPSK Waveform:

For each symbol:

Compute the waveform using:

cos(2πft+ϕ) for the in-phase (real) part

sin(2πft+ϕ) for the quadrature (imaginary) part

Combine both to form a complex baseband signal.

Append to the full signal array.

Plot the Signal:

In-phase component vs. time

Quadrature component vs. time

Real part of the full QPSK waveform

Annotate symbol positions with corresponding binary values.

# PROGRAM

import numpy as np

import matplotlib.pyplot as plt

num_symbols = 10 # Number of symbols (reduced for clarity in the plot)

T = 1.0 # Symbol period

fs = 100.0 # Sampling frequency

t = np.arange(0, T, 1/fs) # Time vector for one symbol

bits = np.random.randint(0, 2, num_symbols * 2) # Two bits per QPSK symbol

symbols = 2 * bits[0::2] + bits[1::2] # Map bits to QPSK symbols

qpsk_signal = np.array([])

symbol_times = []

symbol_phases = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}

for i, symbol in enumerate(symbols):

   phase = symbol_phases[symbol]
   
   symbol_time = i * T
   
   qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase)
   
   qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment))
   
   symbol_times.append(symbol_time)

t_total = np.arange(0, num_symbols * T, 1/fs)

plt.figure(figsize=(14, 12))

plt.subplot(3, 1, 1)

plt.plot(t_total, np.real(qpsk_signal), label='In-phase')

for i, symbol_time in enumerate(symbol_times):

  plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
  
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue')

plt.title('QPSK Signal - In-phase Component with Symbols')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.subplot(3, 1, 2)

plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature', color='orange')

for i, symbol_time in enumerate(symbol_times):

  plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
  
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue')

plt.title('QPSK Signal - Quadrature Component with Symbols')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.subplot(3, 1, 3)

plt.plot(t_total, np.real(qpsk_signal), label='Resultant QPSK Waveform', color='green')

for i, symbol_time in enumerate(symbol_times):

  plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
  
plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue')

plt.title('Resultant QPSK Waveform')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# OUTPUT
![image](https://github.com/user-attachments/assets/1d750b0b-901d-4d83-931b-d3c86f7c31c2)

 
# RESULT / CONCLUSIONS

The QPSK (Quadrature Phase Shift Keying) signal was successfully generated using python.

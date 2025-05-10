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

num_symbols = 10  

T = 1.0  

fs = 100.0  

t = np.arange(0, T, 1/fs)


bits = np.random.randint(0, 2, num_symbols * 2)

i_bits = bits[0::2]  # Even-indexed bits

q_bits = bits[1::2]  # Odd-indexed bits


i_values = 2 * i_bits - 1

q_values = 2 * q_bits - 1


i_signal = np.array([])

q_signal = np.array([])

combined_signal = np.array([])

symbol_times = []

for i in range(num_symbols):

    i_carrier = i_values[i] * np.cos(2 * np.pi * t / T)

    q_carrier = q_values[i] * np.sin(2 * np.pi * t / T)

    symbol_times.append(i * T)

    i_signal = np.concatenate((i_signal, i_carrier))

    q_signal = np.concatenate((q_signal, q_carrier))

    combined_signal = np.concatenate((combined_signal, i_carrier + q_carrier))

t_total = np.arange(0, num_symbols * T, 1/fs)



plt.figure(figsize=(14, 9))



plt.subplot(3, 1, 1)

plt.plot(t_total, i_signal, label='In-phase (cos)', color='blue')

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)

    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}', fontsize=12, color='black')

plt.title('In-phase Component (Cosine) - One Bit per Symbol')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.subplot(3, 1, 2)

plt.plot(t_total, q_signal, label='Quadrature (sin)', color='orange')

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)

    plt.text(symbol_time + T/4, 0.8, f'{q_bits[i]}', fontsize=12, color='black')

plt.title('Quadrature Component (Sine) - One Bit per Symbol')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()


plt.subplot(3, 1, 3)

plt.plot(t_total, combined_signal, label='QPSK Signal = I + Q', color='green')

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)

    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}{q_bits[i]}', fontsize=12, color='black')

plt.title('Combined QPSK Waveform')

plt.xlabel('Time')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# OUTPUT
![image](https://github.com/user-attachments/assets/29717c5c-573b-4bac-bb34-e4683681dba7)



# RESULT / CONCLUSIONS

The QPSK (Quadrature Phase Shift Keying) signal was successfully generated using python.

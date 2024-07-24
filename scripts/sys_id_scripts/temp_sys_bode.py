import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode

# Parameters
g = 9.82
b = 0.2
l = 1.0

# Transfer function coefficients
num = [1]
den = [1, b / l, g / l]

# Create the transfer function
system = TransferFunction(num, den)
# Generate frequency range
w = np.logspace(-2, 2, 1000)  # Frequency range from 0.01 to 100 rad/s

# Compute Bode plot
w, mag, phase = bode(system, w)
w = w / (2 * np.pi)
# Plot magnitude
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title("Bode Plot - Magnitude")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(which="both", axis="both")

# Plot phase
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.title("Bode Plot - Phase")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [degrees]")
plt.grid(which="both", axis="both")

plt.show()
plt.savefig("media_output/temp_pend_bode.png")

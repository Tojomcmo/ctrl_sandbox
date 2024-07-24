import numpy as np
import matplotlib.pyplot as plt

# Example frequency response data (replace with actual data)
frequencies = np.linspace(0.1, 10, 100)  # Frequency range in radians per second
H_exp = np.sin(frequencies) + 1j * np.cos(
    frequencies
)  # Placeholder for actual complex response data

# Define the order of the numerator and denominator
num_order = 2
den_order = 2

# Construct the matrix A and vector b
omega = 1j * frequencies
A = np.zeros((2 * len(frequencies), num_order + den_order), dtype=np.complex128)
b = np.zeros(2 * len(frequencies), dtype=np.complex128)

# Fill the matrix A and vector b
for i in range(len(frequencies)):
    w = omega[i]
    A[2 * i, : num_order + 1] = [1, w, w**2]  # Real part for numerator coefficients
    A[2 * i, num_order + 1 :] = -H_exp[i] * np.array(
        [w, w**2]
    )  # Real part for denominator coefficients
    b[2 * i] = H_exp[i].real

    A[2 * i + 1, : num_order + 1] = [
        1,
        w,
        w**2,
    ]  # Imaginary part for numerator coefficients
    A[2 * i + 1, num_order + 1 :] = -H_exp[i] * np.array(
        [w, w**2]
    )  # Imaginary part for denominator coefficients
    b[2 * i + 1] = H_exp[i].imag

# Solve for the coefficients using the pseudo-inverse
coeffs = np.linalg.pinv(A) @ b

# Extract the numerator and denominator coefficients
num_coeffs = coeffs[: num_order + 1]
den_coeffs = np.concatenate(([1], coeffs[num_order + 1 :]))

print("Numerator coefficients:", num_coeffs)
print("Denominator coefficients:", den_coeffs)


# Define the transfer function model
def transfer_function_model(num_coeffs, den_coeffs, s):
    num = np.polyval(num_coeffs[::-1], s)
    den = np.polyval(den_coeffs[::-1], s)
    return num / den


# Compute the fitted transfer function response
H_fit = transfer_function_model(num_coeffs, den_coeffs, omega)

# Plot the original and fitted responses
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(H_exp)), label="Actual Response")
plt.plot(
    frequencies, 20 * np.log10(np.abs(H_fit)), label="Fitted Response", linestyle="--"
)
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(frequencies, np.angle(H_exp), label="Actual Response")
plt.plot(frequencies, np.angle(H_fit), label="Fitted Response", linestyle="--")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Phase (radians)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
plt.savefig("media_output/ls_fit_bode.png")

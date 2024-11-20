from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Example data
time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
temperature = np.array([20, 35, 50, 60, 68, 73, 77, 80, 82, 83])

# Exponential growth model
def exponential_growth(t, T_steady, T_0, k):
    return T_steady - (T_steady - T_0) * np.exp(-k * t)

# Logarithmic growth model
def logarithmic_growth(t, a, b):
    return a * np.log(t) + b

# Fit the models
exp_popt, exp_pcov = curve_fit(exponential_growth, time, temperature, p0=[85, 20, 0.1])
log_popt, log_pcov = curve_fit(logarithmic_growth, time, temperature)

# Generate fitted data
time_fit = np.linspace(min(time), max(time), 100)
exp_fit = exponential_growth(time_fit, *exp_popt)
log_fit = logarithmic_growth(time_fit, *log_popt)

# Plot the data and fits
plt.figure(figsize=(10, 6))
plt.scatter(time, temperature, label='Data', color='red')
plt.plot(time_fit, exp_fit, label=f'Exponential Fit: {exp_popt[0]:.2f} - ({exp_popt[0]:.2f} - {exp_popt[1]:.2f}) * exp(-{exp_popt[2]:.2f} * t)', color='blue')
plt.plot(time_fit, log_fit, label=f'Logarithmic Fit: {log_popt[0]:.2f} * log(t) + {log_popt[1]:.2f}', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
#%%
import os
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def get_exif_time(image_path):
    """Extract the DateTime metadata from an image."""
    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data:
        for tag, value in exif_data.items():
            if ExifTags.TAGS.get(tag) == "DateTime":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    return None

def process_image(file_path):
    """Process a thermal image and extract temperature data."""
    img = Image.open(file_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    # map pixel values to temperatures
    t_min = 14.7
    t_max = 300
    temp_array = t_min + (img_array / 255) * (t_max - t_min)
    rows, cols = temp_array.shape

    # make all pixels outside of the center equal to 0
    temp_array[:75, :] = 0
    temp_array[:, :130] = 0
    # temp_array[50:, :] = 0

    # Find the highest temperature and its position
    max_temp = np.max(temp_array)
    
    max_positions = np.argwhere(temp_array == max_temp)
    
    # Find the top 20 highest temperatures and their positions
    top_n = 70
    flat_indices = np.argpartition(temp_array.flatten(), -top_n)[-top_n:]
    top_positions = np.column_stack(np.unravel_index(flat_indices, temp_array.shape))

    # Initialize variables to track the best line
    best_line_sum = -np.inf
    best_line = None
    best_line_coords = (0, 0, 0)

    line_length = 100

    # Check each position to find the best horizontal line
    for pos in top_positions:
        row, col = pos
        start_col = max(0, col - line_length)
        end_col = min(cols, col + line_length + 1)
        line = temp_array[row, start_col:end_col]
        line_sum = np.sum(line)

        if line_sum > best_line_sum:
            best_line_sum = line_sum
            best_line = line
            best_line_coords = (row, start_col, end_col)

    # Use the best line found
    center_row, start_col, end_col = best_line_coords
    horizontal_line = best_line

    # TroubleshootLinePosition(temp_array, center_row, start_col, end_col)

    return {
        "max_temp": max_temp,
        "horizontal_line": horizontal_line,
        "line_coords": (center_row, start_col, end_col),
        "time": get_exif_time(file_path),
    }

def TroubleshootLinePosition(temp_array, center_row, start_col, end_col):
    '''plot the temperature profile with the horizontal line overlayed'''

    plt.figure(figsize=(14, 10))
    plt.imshow(temp_array, cmap="hot", vmin=15, vmax=300)
    plt.plot([start_col, end_col], [center_row, center_row], color="blue", linewidth=3)
    plt.xlabel("Column Index", fontsize=20)
    plt.ylabel("Row Index", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(label="Temperature (°C)")
    plt.tight_layout()
    plt.show()

def process_folder(folder_path, reference_time):
    """Process all thermal images in a folder."""
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, file_name)
            results.append(process_image(file_path))

    # Extract temperature profiles and calculate time deltas
    temp_profiles = {}
    for res in results:
        if res["time"] is not None:
            seconds_since_reference = (res["time"] - reference_time).total_seconds()
            temp_profiles[f"{int(seconds_since_reference)} seconds"] = res["horizontal_line"]

    # Create a DataFrame
    temp_profiles_df = pd.DataFrame(temp_profiles)

    return temp_profiles_df

def PlotThermalTraces():
    '''plot the temperature profiles'''

    temperature_profiles = pd.read_csv(csvExportPath)
    plt.figure(figsize=(14, 10))

    # Generate a colormap based on the number of samples
    colormap = plt.cm.viridis
    colors = colormap(np.linspace(0, 1, len(temperature_profiles.columns)))

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    peak_temps = []
    peak_temp_errors = []
    # Iterate over each column to plot the temperature profiles
    for i, column in enumerate(temperature_profiles.columns[:13]):
        temperatures = temperature_profiles[column]
        x_data = range(len(temperatures))

        # Exclude saturated points
        valid_indices = temperatures < 300
        x_data_valid = np.array(x_data)[valid_indices]
        temperatures_valid = temperatures[valid_indices]

        # Fit a horizontal line plus a Gaussian to the profile
        initial_guess = [np.max(temperatures_valid), np.argmax(temperatures_valid), 10]
        popt, pcov = curve_fit(gaussian, x_data_valid, temperatures_valid, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))

        # Plot the original profile
        plt.plot(
            x_data, 
            temperatures,
            color=colors[i],  # Color based on the sample (time)
            linewidth=3,  # Width of the line
            label=f"{column}",
            alpha=0.7,
        )

        # Plot the fitted profile as a dotted line
        fitted_profile = gaussian(np.array(x_data), *popt)
        plt.plot(
            x_data, 
            fitted_profile,
            color=colors[i],  # Same color as the original profile
            linestyle='dotted',
            linewidth=2,
            alpha=0.7,
        )

        # Store the peak temperature from the fit and its error
        peak_temps.append(popt[0])
        peak_temp_errors.append(perr[0])

    plt.xlabel("Pixel Index", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.legend(title="time (s)", loc="upper left", fontsize=14, title_fontsize=14, bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Plot Max T vs time with fits
    plt.figure(figsize=(14, 10))
    max_temps_times = [int(col.split()[0]) for col in temperature_profiles.columns[:13]]
    plt.errorbar(max_temps_times, peak_temps, yerr=peak_temp_errors, fmt='o', color="red", label="Peak Temp", zorder=10)

    # Fit peak temps
    def exponential_growth(t, T_steady, T_0, k):
        return T_steady - (T_steady - T_0) * np.exp(-k * t)
    
    def exponential_decay(t, T_steady, T_0, k):
        return T_0 + (T_steady - T_0) * np.exp(-k * t)

    exp_popt, exp_pcov = curve_fit(exponential_growth, max_temps_times, peak_temps, p0=[85, 20, 0.1])
    time_fit = np.linspace(min(max_temps_times), max(max_temps_times), 100)
    exp_fit = exponential_growth(time_fit, *exp_popt)

    plt.plot(time_fit, exp_fit, label=f'Exponential Fit: {exp_popt[0]:.2f} - ({exp_popt[0]:.2f} - {exp_popt[1]:.2f}) * exp(-{exp_popt[2]:.2f} * t)', color='blue', linewidth=3)

    plt.errorbar(max_temps_times, peak_temps, yerr=peak_temp_errors, fmt='o', color="red", label="Peak Temp", zorder=10, markersize=10)

    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Consolidate profiles and fits of columns [13:] into one figure
    plt.figure(figsize=(14, 10))
    peak_temps_13 = []
    peak_temp_errors_13 = []
    for i, column in enumerate(temperature_profiles.columns[13:], start=13):
        temperatures = temperature_profiles[column]
        x_data = range(len(temperatures))

        # Exclude saturated points
        valid_indices = temperatures < 300
        x_data_valid = np.array(x_data)[valid_indices]
        temperatures_valid = temperatures[valid_indices]

        # Fit a horizontal line plus a Gaussian to the profile
        initial_guess = [np.max(temperatures_valid), np.argmax(temperatures_valid), 10]
        popt, pcov = curve_fit(gaussian, x_data_valid, temperatures_valid, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))

        # Plot the original profile
        plt.plot(
            x_data, 
            temperatures,
            color=colors[i % len(colors)],  # Color based on the sample (time)
            linewidth=3,  # Width of the line
            label=f"{column}",
            alpha=0.7,
        )

        # Plot the fitted profile as a dotted line
        fitted_profile = gaussian(np.array(x_data), *popt)
        plt.plot(
            x_data, 
            fitted_profile,
            color=colors[i % len(colors)],  # Same color as the original profile
            linestyle='dotted',
            linewidth=2,
            alpha=0.7,
        )

        # Store the peak temperature from the fit and its error
        peak_temps_13.append(popt[0])
        peak_temp_errors_13.append(perr[0])

    plt.xlabel("Pixel Index", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.legend(title="time (s)", loc="upper left", fontsize=14, title_fontsize=14, bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Plot Max T vs time with fits for columns [13:]
    plt.figure(figsize=(14, 10))
    max_temps_times_13 = [int(col.split()[0]) for col in temperature_profiles.columns[13:]]
    plt.errorbar(max_temps_times_13, peak_temps_13, yerr=peak_temp_errors_13, fmt='o', color="red", label="Peak Temp", zorder=10)

    # Fit peak temps
    exp_popt_13, exp_pcov_13 = curve_fit(exponential_decay, max_temps_times_13, peak_temps_13, p0=[85, 20, 0.1])
    time_fit_13 = np.linspace(min(max_temps_times_13), max(max_temps_times_13), 100)
    exp_fit_13 = exponential_decay(time_fit_13, *exp_popt_13)

    plt.plot(time_fit_13, exp_fit_13, label=f'Exponential Fit: {exp_popt_13[0]:.2f} - ({exp_popt_13[0]:.2f} - {exp_popt_13[1]:.2f}) * exp(-{exp_popt_13[2]:.2f} * t)', color='blue', linewidth=3)

    plt.errorbar(max_temps_times_13, peak_temps_13, yerr=peak_temp_errors_13, fmt='o', color="red", label="Peak Temp", zorder=10, markersize=10)

    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()



##########
## MAIN ##
##########

folder_path = "Nov19"
reference_time = datetime.strptime("2024-11-19 14:19:14", "%Y-%m-%d %H:%M:%S")

## Process the thermal image data and export temperatures to a new CSV file
temp_profiles_df = process_folder(folder_path, reference_time)
csvExportPath = "exports/temperature_profiles.csv"
temp_profiles_df.to_csv(csvExportPath, index=False)

#%%
## plot the temperature profiles
PlotThermalTraces()
#%%

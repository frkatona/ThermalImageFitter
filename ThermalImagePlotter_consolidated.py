# pixels to cm for line length = 5cm
   # maybe just for the plots since may not be important for the pics
# consolidate graphs (SI)
# produce a version with only a handful of representative datasets (main paper)
# consolidate residuals (SI)

# color red/blue
# intervals to 0.5 cm

import os
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fontsize = 20

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
        "time": get_exif_time(file_path)
    }

def TroubleshootLinePosition(temp_array, center_row, start_col, end_col):
    '''plot the temperature profile with the horizontal line overlayed'''

    plt.figure(figsize=(12, 6))
    plt.imshow(temp_array, cmap="hot", vmin=15, vmax=300)
    plt.plot([start_col, end_col], [center_row, center_row], color="#2233CC", linewidth=3)
    plt.xlabel("Column Index", fontsize=fontsize)
    plt.ylabel("Row Index", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
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

    selected_columns = ["5 seconds", "15 seconds", "30 seconds", "60 seconds", "65 seconds", "90 seconds", "120 seconds"]
    temperature_profiles = pd.read_csv(csvExportPath)#, usecols=selected_columns)
    plt.figure(figsize=(18, 12))

    # Generate a colormap based on the number of samples
    cutoff_index = 12 #selected_columns.index("60 seconds")
    colors = ['#CC3322' if i <= cutoff_index else '#2233CC' for i in range(len(temperature_profiles.columns))]

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    ## Plot temperature profiles
    peak_temps = []
    peak_temp_errors = []
    for i, column in enumerate(temperature_profiles.columns):
        temperatures = temperature_profiles[column]
        x_data = np.linspace(-2.5, 2.5, len(temperatures))  # Map pixel indices to cm with center at 0 cm

        # Exclude saturated points
        valid_indices = temperatures < 270 if i <= cutoff_index else temperatures < 300
        x_data_valid = x_data[valid_indices]
        temperatures_valid = temperatures[valid_indices]

        # Fit a horizontal line plus a Gaussian to the profile
        initial_guess = [np.max(temperatures_valid), x_data_valid[np.argmax(temperatures_valid)], 1]
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
        fitted_profile = gaussian(x_data, *popt)
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

    plt.xlabel("distance from beam center (cm)", fontsize=fontsize)
    plt.ylabel("Temperature (°C)", fontsize=fontsize)
    plt.axvline(-1.5, color='black', linewidth=1, linestyle='dashed')
    plt.axvline(1.5, color='black', linewidth=1, linestyle='dashed')
    plt.legend(title="time (s)", loc="upper left", fontsize=fontsize, bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    # Plot Max T vs time with fits
    plt.figure(figsize=(12, 6))
    max_temps_times = [int(col.split()[0]) for col in temperature_profiles.columns]

    plt.errorbar(max_temps_times, peak_temps, yerr=peak_temp_errors, fmt='o', color="#CC3322", label="Peak Temp", zorder=10)
    
    # Fit peak temps
    def exponential_growth(t, T_steady, T_0, k):
        return T_steady - (T_steady - T_0) * np.exp(-k * t)

    def exponential_decay(t, T_steady, T_0, k):
        return T_0 + (T_steady - T_0) * np.exp(-k * t)

    # Fit for initial phase (up to 60 seconds)
    exp_popt_heating, exp_pcov = curve_fit(exponential_growth, max_temps_times[:cutoff_index+1], peak_temps[:cutoff_index+1], p0=[85, 20, 0.1])
    time_fit_heating = np.linspace(min(max_temps_times[:cutoff_index+1]), max(max_temps_times[:cutoff_index+1]), 100)
    exp_fit_heating = exponential_growth(time_fit_heating, *exp_popt_heating)

    plt.plot(time_fit_heating, exp_fit_heating, label=f'Exponential Growth Fit: {exp_popt_heating[0]:.2f} - ({exp_popt_heating[0]:.2f} - {exp_popt_heating[1]:.2f}) * exp(-{exp_popt_heating[2]:.2f} * t)', color='#2233CC', linewidth=3)

    # Fit for later phase (after 60 seconds)
    exp_popt_cooling, exp_pcov_13 = curve_fit(exponential_decay, max_temps_times[cutoff_index:], peak_temps[cutoff_index:], p0=[18000, 60, 0.1])
    time_fit_cooling = np.linspace(min(max_temps_times[cutoff_index:]), max(max_temps_times[cutoff_index:]), 100)
    exp_fit_cooling = exponential_decay(time_fit_cooling, *exp_popt_cooling)

    plt.plot(time_fit_cooling, exp_fit_cooling, label=f'Exponential Decay Fit: {exp_popt_cooling[0]:.2f} - ({exp_popt_cooling[0]:.2f} - {exp_popt_cooling[1]:.2f}) * exp(-{exp_popt_cooling[2]:.2f} * t)', color='#2233CC', linewidth=3)

    plt.errorbar(max_temps_times, peak_temps, yerr=peak_temp_errors, fmt='o', color='#CC3322', label="Peak Temp", zorder=10, markersize=10)
    plt.xlabel("time (s)", fontsize=fontsize)
    plt.ylabel("temperature (°C)", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

##########
## MAIN ##
##########


# # Extract reference time from images
# folder_path = "Nov19"
# first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
# time_0 = get_exif_time(first_image_path)
# time_offset = 1 # seconds before first image
# reference_time = time_0 - pd.to_timedelta(time_offset, unit='s')

## Process the thermal image data and export temperatures to a new CSV file
# temp_profiles_df = process_folder(folder_path, reference_time)
csvExportPath = "exports/temperature_profiles.csv"
# temp_profiles_df.to_csv(csvExportPath, index=False)

#%%
## plot the temperature profiles
PlotThermalTraces()
#%%

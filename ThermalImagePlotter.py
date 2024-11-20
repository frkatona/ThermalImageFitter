#%%
import os
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
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

    #! instead of RoI, turn pixels with x < 120 or y < 70 to 0

    rows, cols = temp_array.shape
    # temp_array[:70, :] = 0
    temp_array[:, 500:] = 0


    #! allow the line's bounds to go past this quadrant, just not its center

    # Find the highest temperature and its position
    max_temp = np.max(temp_array)
    
    if max_temp == 300:
        # Find all positions with temperature 300
        max_positions = np.argwhere(temp_array == 300)
        
        # Initialize variables to track the best line
        best_line_sum = -np.inf
        best_line = None
        best_line_coords = (0, 0, 0)
        
        # Check each position to find the best horizontal line
        for pos in max_positions:
            row, col = pos
            start_col = max(0, col - 55)
            end_col = min(cols, col + 55 + 1)
            line = temp_array[row, start_col:end_col]
            line_sum = np.sum(line)
            
            if line_sum > best_line_sum:
                best_line_sum = line_sum
                best_line = line
                best_line_coords = (row, start_col, end_col)
        
        # Use the best line found
        center_row, start_col, end_col = best_line_coords
        horizontal_line = best_line
    else:
        # Find the highest temperature and its position
        max_pos = np.unravel_index(np.argmax(temp_array), temp_array.shape)
        max_pos_full = (max_pos[0] + rows // 2, max_pos[1] + cols // 2)

        # Extract a horizontal line of temperatures (31 pixels centered on max position)
        center_row = max_pos_full[0]
        center_col = max_pos_full[1]
        half_window = 55
        start_col = max(0, center_col - half_window)
        end_col = min(cols, center_col + half_window + 1)
        horizontal_line = temp_array[center_row, start_col:end_col]

    TroubleshootLinePosition(temp_array, center_row, start_col, end_col)

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

def PlotTemperatureProfiles():
    # Load the temperature profiles from the CSV file
    temperature_profiles = pd.read_csv(csvExportPath)

    # Create a figure and overlay the profiles
    plt.figure(figsize=(14, 10))

    # Generate a colormap based on the number of samples
    colormap = plt.cm.viridis
    colors = colormap(np.linspace(0, 1, len(temperature_profiles.columns)))

    # Iterate over each column to plot the temperature profiles
    for i, column in enumerate(temperature_profiles.columns):
        temperatures = temperature_profiles[column]
        smoothed_temperatures = gaussian_filter1d(temperatures, sigma=2)  # Smooth the line
        plt.plot(
            range(len(smoothed_temperatures)), 
            smoothed_temperatures,
            color=colors[i],  # Color based on the sample (time)
            linewidth=3,  # Width of the line
            label=f"{column}",
            alpha=0.7,
        )
    plt.xlabel("Pixel Index", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.legend(title="time (s)", loc="upper left", fontsize=14, title_fontsize=14, bbox_to_anchor=(1.05, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Plot Max T vs time with fits
    plt.figure(figsize=(14, 10))
    max_temps = temperature_profiles.apply(np.max, axis=0)
    max_temps_times = [int(col.split()[0]) for col in max_temps.index]
    plt.scatter(max_temps_times, max_temps, color="red", s=100, label="Max Temp", zorder=10)

    # # fits to max T vs time
    # max_temps_times_log = np.log(max_temps_times[:4])
    # max_temps_log = np.log(max_temps[:4])
    # max_temps_times_exp = max_temps_times[13:]
    # max_temps_exp = max_temps[13:]
    # p_log = np.polyfit(max_temps_times_log, max_temps_log, 1)
    # p_exp = np.polyfit(max_temps_times_exp, max_temps_exp, 1)
    # plt.plot(max_temps_times, np.exp(np.polyval(p_log, np.log(max_temps_times))), color="blue", label="Log Growth Fit")
    # plt.plot(max_temps_times, np.polyval(p_exp, max_temps_times), color="green", label="Exponential Decay Fit")

    # Render figures
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
PlotTemperatureProfiles()
#%%

import math
from datetime import datetime, timedelta

def solar_position(latitude, longitude, local_time):
    """
    Calculate the solar elevation and azimuth using the formulas from slides 38 to 55.
    
    Parameters:
        latitude (float): Latitude of the location in degrees.
        longitude (float): Longitude of the location in degrees.
        local_time (datetime): Local datetime of the observation.
    
    Returns:
        tuple: (elevation, azimuth) in degrees.
    """
    # Convert local time to UTC
    utc_offset = 1  # Brussels is UTC+1 in standard time (consider DST if needed)
    utc_time = local_time - timedelta(hours=utc_offset)
    
    # Day of the year
    day_of_year = utc_time.timetuple().tm_yday
    
    # Declination angle (δ)
    declination = -23.45 * math.cos(math.radians((360 / 365) * (day_of_year + 10)))
    
    # Local Standard Time Meridian (LSTM)
    LSTM = 15 * utc_offset
    
    # Equation of Time (EoT)
    B = math.radians((360 / 365) * (day_of_year - 81))
    EoT = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    
    # Time Correction Factor (TC)
    TC = 4 * (longitude - LSTM) + EoT
    
    # Local Solar Time (LST)
    LST = local_time.hour + local_time.minute / 60 + local_time.second / 3600 + TC / 60
    
    # Hour Angle (HRA)
    HRA = 15 * (LST - 12)
    
    # Convert angles to radians
    latitude_rad = math.radians(latitude)
    declination_rad = math.radians(declination)
    HRA_rad = math.radians(HRA)
    
    # Elevation angle (α)
    elevation = math.degrees(math.asin(math.sin(latitude_rad) * math.sin(declination_rad) +
                                       math.cos(latitude_rad) * math.cos(declination_rad) * math.cos(HRA_rad)))
    
    # Azimuth angle (Θ)
    azimuth = math.degrees(math.acos((math.sin(declination_rad) * math.cos(latitude_rad) -
                                      math.cos(declination_rad) * math.sin(latitude_rad) * math.cos(HRA_rad)) /
                                      math.cos(math.radians(elevation))))
    
    # Adjust azimuth angle based on hour angle
    if HRA > 0:
        azimuth = 360 - azimuth
    
    return elevation, azimuth


def solar_position_at_zenith(latitude, longitude, local_time):
    """
    Calculate the solar elevation and azimuth at solar noon.
    
    Parameters:
        latitude (float): Latitude of the location in degrees.
        longitude (float): Longitude of the location in degrees.
        local_time (datetime): Local datetime of the observation.
    
    Returns:
        tuple: (elevation, azimuth) in degrees.
    """
    utc_offset = 1  # Brussels is UTC+1 in standard time (consider DST if needed)
    utc_time = local_time - timedelta(hours=utc_offset)
    
    # Day of the year
    day_of_year = utc_time.timetuple().tm_yday
    
    # Declination angle (δ)
    declination = -23.45 * math.cos(math.radians((360 / 365) * (day_of_year + 10)))
    
    
    # Hour Angle (HRA)
    HRA = 0 # Solar noon
    
    # Convert angles to radians
    latitude_rad = math.radians(latitude)
    declination_rad = math.radians(declination)
    HRA_rad = math.radians(HRA)
    
    # Elevation angle (α)
    elevation = math.degrees(math.asin(math.sin(latitude_rad) * math.sin(declination_rad) +
                                       math.cos(latitude_rad) * math.cos(declination_rad) * math.cos(HRA_rad)))
    

    
    return elevation







# Example usage
latitude = 50.8503  # Brussels latitude
longitude = 4.3517  # Brussels longitude
local_time = datetime(2024, 6, 21, 12, 0)  # Example date and time

elevation, azimuth = solar_position(latitude, longitude, local_time)
print(f"Elevation: {elevation:.2f}°")
print(f"Azimuth: {azimuth:.2f}°")

print("Zenith")
elevation = solar_position_at_zenith(latitude, longitude, local_time)
print(f"Elevation: {elevation:.2f}°")



# graphe elevation vs  jour de l'année

import matplotlib.pyplot as plt
import numpy

# Latitude and longitude of Brussels
latitude = 50.8503
longitude = 4.3517

# Create an array of days from 1 to 365
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# Calculate solar elevation for each day
elevations = []

for month in months:
    if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        days = numpy.arange(1, 32)
    elif month == 2:
        days = numpy.arange(1, 29)
    else:
        days = numpy.arange(1, 31)
    for day in days:
        local_time = datetime(2024, month, day, 12, 0)
        elevation, azimuth = solar_position(latitude, longitude, local_time)
        elevations.append(elevation)
x = 12
days = numpy.arange(1, 366)
# Plot the solar elevation vs. day of the year
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(days, elevations, linewidth=1)
plt.xlabel("Day of the year", fontsize=x)
plt.ylabel("Solar Elevation [°]", fontsize=x)
plt.savefig('Output/elevation_vs_day.svg')


# graphe elevation_at_zenith vs  jour de l'année

elevation_at_zenith = []

for month in months:
    if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        days = numpy.arange(1, 32)
    elif month == 2:
        days = numpy.arange(1, 29)
    else:
        days = numpy.arange(1, 31)
    for day in days:
        local_time = datetime(2024, month, day, 12, 0)
        elevation = 90 - solar_position_at_zenith(latitude, longitude, local_time)
        elevation_at_zenith.append(elevation)
x = 12
days = numpy.arange(1, 366)
# Plot the solar elevation vs. day of the year
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(days, elevation_at_zenith, linewidth=1)
plt.xlabel("Day of the year", fontsize=x)
plt.ylabel("Incidance Angle at zenith [°]", fontsize=x)
plt.savefig('Output/incidence_at_zenith_vs_day.svg')

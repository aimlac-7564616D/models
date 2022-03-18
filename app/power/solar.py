#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in API forecast and output hourly solar power production predictions 
for the next 24 between 23:00 and 23:00
Based on SLIMJAB
"""
from dataclasses import dataclass
from datetime import datetime
from flask import current_app
from scipy.integrate import quad
from scipy.interpolate import interp1d

import numpy as np
import os
import pandas as pd
import pvlib as pv


@dataclass
class SolarArray:
    """Class to represent a solar panel array.

    Attributes:
        area (float): Total solar array area in m^2.
        tilt (float): Angle in degrees that panels are inclined facing South.
        base_eff (float): Baseline efficiency of solar panels in array.
        pmpp (float): Temperature coefficient of panels in % per degree celsius.
        pmax (float): Maximum total output of solar array in Watts.

    """

    array_area: float
    panel_tilt: float
    base_efficiency: float
    temperature_coeff: float
    max_output: float


def temperature_efficiency(
    base_eff: float, temp_coef: float, forecast: pd.DataFrame
) -> float:
    """Function to calculate the effective panel efficiency at a particular temperature.

    Args:
        base_eff (float): Base efficiency.
        temp_coef (float): Temperature coefficient in % per degree celsius.
        forecast (pd.DataFrame): Forcast DataFrame.

    Returns:
        float: solar panel efficiency at the given temperature.
    """
    temperature = forecast["screenTemperature"]
    temperature_diff = temperature - 25
    eff_change = temperature_diff * temp_coef
    return base_eff + eff_change


def weather_factor(forecast: pd.DataFrame) -> pd.Series:
    """Returns the weather efficiency factor for the forcast.

    Returns weather factor efficiency multiplier based on MetOffice API
    weather type for a one hour timestep.

    Args:
        forecast (pd.DataFrame): Forcast dataframe.

    Returns:
        pd.Series: Weather efficiency factor for the provided forcast.
    """
    weather = forecast["significantWeatherCode"]
    weather_list = []
    no_change = [0, 1]
    med_change = [2, 3, 5, 8]
    heavy_change = [6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 28, 29, 30]
    largest_change = [19, 20, 21, 22, 23, 24, 25, 26, 27, 4]
    for wtype in weather:
        if wtype in no_change:
            weather_list.append(1.0)
        if wtype in med_change:
            weather_list.append(0.5)
        if wtype in heavy_change:
            weather_list.append(0.1)
        elif wtype in largest_change:
            weather_list.append(0.0)
    return pd.Series(weather_list)


def get_incident_power(
    forecast: pd.DataFrame, location: pv.location.Location, tilt: float, area: float
) -> pd.Series:
    """Returns the clear-sky power incident on solar panel array.

    Returns clear-sky power from Sun incident on a solar panel array of a
    given area and tilt (South facing) at a given location for the times
    provided by MetOffice API forecast output.

    Args:
        forecast (pd.DataFrame): Forcast dataframe.
        location (pv.location.Location): Location of solar array.
        tilt (float): Angle in degrees that panels are inclined facing South.
        area (float): Total solar array area in m^2.

    Returns:
        pd.Series: Clear-sky power incident on solar array for each forecast timestep.
    """
    times = pd.DatetimeIndex(
        forecast["time"]
    )  # define times to model sun position / insolation from
    solar_pos = location.get_solarposition(
        times
    )  # calculate solar position at defined location on defined times
    clearsky = location.get_clearsky(
        times
    )  # calculate insolation at location for date range
    ghi = clearsky.ghi  # calculate insolation incident on tilted solar panel
    flux_density = (
        ghi * np.sin(np.deg2rad(solar_pos["apparent_elevation"]) + np.deg2rad(tilt))
    ) / np.sin(np.deg2rad(solar_pos["apparent_elevation"]))
    return flux_density * area  # Watts #calculate power of whole array


def get_total_efficiency(
    forecast: pd.DataFrame, base_eff: float, pmpp: float
) -> pd.Series:
    """Returns total efficiency of Solar panels for each forecast step.

    Returns the total efficiency of Solar panels for each forecast time step
    depending on temperature and weather condition factors.

    Args:
        forecast (pd.DataFrame): Forcast dataframe.
        base_eff (float): Baseline efficiency of solar panels.
        pmpp (float): Temperature coefficient of solar panels (% per degree Celsius).

    Returns:
        pd.series: Total Solar panel efficiency at each forecast timestep.
    """
    times = pd.DatetimeIndex(forecast["time"])
    efficiency_temperature_mod = temperature_efficiency(
        base_eff, pmpp, forecast
    )  # effiency change from temperature
    weather = weather_factor(forecast)  # weather factor efficency multiplier
    total_eff = efficiency_temperature_mod * np.asarray(weather)  # total efficiency
    total_eff.index = times  # change index to datetimes consistent with forecast
    return total_eff


def get_generated_power(
    incident_power: pd.Series,
    total_efficiency: pd.Series,
    datetimes: pd.DatetimeIndex,
    max_array_output: float,
) -> pd.DataFrame:
    """Returns net generated power of solar array from incident power and panel
    efficiency.

    Returns a series object containing the solar array predicted output for each
    timestep from an incident power series object and a panel efficiency series in
    Watts.

    Args:
        incident_power (pd.Series): Clear-sky power incident on solar array for each
        forecast timestep.
        total_efficiency(pd.Series): Total Solar panel efficiency at each forecast
        timestep.
        datetimes (pd.DatetimeIndex): Array of Datetimes corresponding to the start
        of each forecast timestep.
        max_array_output (float): Maximum total output of solar array in Watts.

    Returns:
        pd.Series: Predicted solar array output for each timestep of forecast in Watts.
    """
    output_power = incident_power * total_efficiency  # Watts
    output_power[
        output_power > max_array_output
    ] = max_array_output  # cap power output at max power output of array
    hours_since_23 = np.linspace(0, 24, 25)
    power_curve = interp1d(
        hours_since_23, output_power
    )  # interpolate power into function of time
    generated_power = []
    for i in range(len(hours_since_23) - 1):
        lower_limit = hours_since_23[i]
        upper_limit = hours_since_23[i + 1]
        integral = quad(
            power_curve, lower_limit, upper_limit
        )  # integrate hourly intervals
        generated_power.append(integral[0])  # Watts/hr
    generated_power.append(0.0)  # 23:00 - 00:00 interval needs a value for array shapes

    return pd.DataFrame(data={"time": datetimes, "SolarPower": generated_power})


def predict_solar(
    forecast: pd.DataFrame, location: pv.location.Location, solar_array: SolarArray
) -> pd.DataFrame:
    """Returns predicted solar array output for a given forecast, location and solar
    array parameters.

    Wrapper function that returns predicted solar array output for each timestep of
    a given forecast at a defined location and for a set of solar array parameters.

    Args:
        forecast (pd.DataFrame): Forcast dataframe.
        location (pv.location.Location): Location object of solar array.
        solar_array (SolarArray): Object describing solar array properties.

    Returns:
        pd.DataFrame: Predicted solar array output for each timestep of forecast in Watts.
    """
    forecast_datetimes = pd.DatetimeIndex(
        forecast["time"]
    )  # create array of forecast step DatetimeIndicies
    power_incident = get_incident_power(
        forecast, location, solar_array.panel_tilt, solar_array.array_area
    )  # calculate power incident on array
    total_efficiency = get_total_efficiency(
        forecast, solar_array.base_efficiency, solar_array.temperature_coeff
    )  # calculate total efficiency based on temp and weather conditions
    power_generated = get_generated_power(
        power_incident, total_efficiency, forecast_datetimes, solar_array.max_output
    )  # calculate the power generated by the solar array for each forecast timestep
    return power_generated


def cut_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cut a dataframe on the DateTime column into intervals of 23:00 - 23:00.

    Cuts a given Pandas DataFrame via a string DateTime column into an interval
    of 23:00 - 23:00.

    Args:
        frame (pd.DataFrame): Pandas dataframe to be cut. Must contain a DataTime
            column.

    Returns: pd.DataFrame: Pandas dataframe cut into intervals of 23:00 - 23:00
    """
    datetime_format = current_app.config["DATETIME_FORMAT"]
    hrs = []
    for i in range(len(frame["time"])):
        hrs.append(datetime.strptime(frame["time"][i], datetime_format).hour)
    idxs = np.where(np.asarray(hrs) == 23)[0]
    start = idxs[0]
    end = idxs[1]
    return frame.truncate(start, end)


def get_solar_prediction(forecast: pd.DataFrame) -> pd.DataFrame:
    """Get the predicted Solar Panel output.

    Returns the predicted solar array output in W for next 23:00 - 23:00 interval,
    in 60 min steps.

    Args:
        forecast (dict): Forcast dataframe.

    Returns: pd.DataFrame: The predicted solar energy output of solar array over the
    next 24 hours.
    """
    forecast = cut_frame(forecast)

    # set panel / location parameters
    latitude = current_app.config["LATITUDE"]
    longitude = current_app.config["LONGITUDE"]
    timezone = current_app.config["TIMEZONE"]
    altitude = current_app.config["ALTITUDE"]
    panel_tilt = current_app.config["PANEL_TILT"]
    array_area = current_app.config["ARRAY_AREA"]
    base_efficiency = current_app.config["BASE_EFFICIENCY"]
    pmpp = current_app.config["PMPP"]
    pmax_array = current_app.config["PMAX_ARRAY"]

    aimlac_location = pv.location.Location(
        float(latitude), float(longitude), tz=timezone, altitude=altitude
    )  # define location of solar panel installation
    aimlac_solar_array = SolarArray(
        array_area, panel_tilt, base_efficiency, pmpp, pmax_array
    )  # create SolarArray object to describe aimlacHQ solar panel array
    predicted_solar_output = predict_solar(
        forecast, aimlac_location, aimlac_solar_array
    )  # calculate predicted solar output from forecast, location and array parameters
    predicted_solar_output["SolarPower"] /= 1000.0  # convert to kW

    return predicted_solar_output

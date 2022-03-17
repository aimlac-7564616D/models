# -*- coding: utf-8 -*-
"""
Read in API forecast and output hourly wind power production predictions 
for the next 24 between 23:00 and 23:00
Based on SLIMJAB
"""
from app.models import wind_model

from datetime import datetime
from flask import current_app
from typing import Union

import numpy as np
import os
import pandas as pd
import pickle as p

def get_wind_speed(forecast: pd.DataFrame) -> pd.DataFrame:
    """Unpacks wind speed from forecast.
     Unpacks wind speed in mph from a pd.DataFrame containing forecast data and
     converts it to m/s.
     Args:
        forecast (pd.DataFrame) : MetOffice forecast data.
    Returns:
        pd.DataFrame object containing wind speed and datetime information.
    """
    assert isinstance(forecast, pd.DataFrame)

    return forecast[["time", "windSpeed10m"]].reset_index()


def get_wind_power(windspeed: Union[list, np.ndarray, pd.Series, float],
                   height: Union[int, float] = 10) -> np.ndarray:
    """Converts windspeed to power.
     Converts a windspeed or set of windspeeds at a given height to power
     generated in kW according to the wind model created from the turbine
     datasheets.
     Args:
        windspeed (list, np.ndarray, float  : Windspeed or set of
                   pd.Series)                 windspeeds in m/s.
        height (int, float)                 : Rotor height (default is 10m)
    Returns:
        np.ndarray object corresponding to the power generated (in kW).
    Raises:
        ValueError  : if windspeed < 0
        ValueError  : if height < 0
    """

    n_turbines = 6  # Number of active turbines
    hellman_exp = 0.34  # Hellman exponent for static air above inhabited areas
    assert isinstance(windspeed, (list, np.ndarray, float, pd.Series))
    assert isinstance(height, (int, float))

    if isinstance(windspeed, (list, float, pd.Series)):
        windspeed = np.asarray(windspeed)
    assert np.all(windspeed >= 0)  # Check all windspeeds are non-negative
    assert height >= 0

    correction = (height / 10)**hellman_exp  # Terrain/height correction

    windspeed *= correction
    windspeed[windspeed >= 30] = 30  # Truncate excess speeds
    return n_turbines * wind_model(windspeed)


def cut_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cut a dataframe on the DateTime column into intervals of 23:00 - 23:00.

    Cuts a given Pandas DataFrame via a string DateTime column into an interval
    of 23:00 - 23:00.

    Args:
        frame (pd.DataFrame): Pandas dataframe to be cut. Must contain a DataTime
            column.

    Returns: pd.DataFrame: Pandas dataframe cut into intervals of 23:00 - 23:00
    """
    datetime_format = current_app.config['DATETIME_FORMAT']
    hrs = []
    for i in range(len(frame["time"])):
        hrs.append(datetime.strptime(frame["time"][i], datetime_format).hour)
    idxs = np.where(np.asarray(hrs) == 23)[0]
    start = idxs[0]
    end = idxs[1] 
    return frame.truncate(start, end)


def get_wind_prediction(forecast: pd.DataFrame)->pd.DataFrame:
    """Wrapper function to return array of predicted wind power generation for forecast timesteps.
    
    Args:
        forecast (dict): Forcast dataframe.

    Returns:
        wind_report (pd.DataFrame): The wind speed and predicted wind energy output of wind turbines over the 
    next 24 hours, in 30 minute intervals (48 instances).
    """
    altitude = current_app.config['ALTITUDE']
    forecast = cut_frame(forecast)
    wind_speed = get_wind_speed(forecast)
    wind_power = get_wind_power(wind_speed["windSpeed10m"], altitude)
    wind_report = pd.DataFrame(data={"time": wind_speed["time"], "WindSpeed": wind_speed["windSpeed10m"], "WindPower": wind_power})
    return wind_report
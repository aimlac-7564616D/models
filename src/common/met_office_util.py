from datetime import datetime

import numpy as np
import pandas as pd

import src.config as config

def interp_30min(frame: pd.DataFrame) -> pd.DataFrame:
    """Interpolates hourly weather report into 30-min intervals.

    Interpolates hourly weather report into 30-min intervals and attaches
    corresponding DateTime stamps. Forecast between 23:00-23:00 returned as a
    Pandas DataFrame

    Args:
        frame (pd.DataFrame): Met office forecast dataframe.

    Returns: pd.DataFrame: A dataframe containing the forcast variables along
        with datetimes across a 23:00 - 23:00 timespan in 30 minute interpreted
        increments.
    """
    assert "time" in frame.columns, "no timestamp"
    frame.time = pd.to_datetime(frame.time, format=config.DATETIME_FORMAT)
    frame.time = frame.time.map(lambda dt: dt.replace(minute=0, second=0))
    start_dt = frame.time.min().replace(hour=23)
    end_dt = start_dt.replace(day=start_dt.day + 1)
    assert end_dt < frame.time.max(), "not enough data"

    frame = frame.set_index('time')
    frame = frame[start_dt:end_dt]
    frame = frame.asfreq('30min')
    
    disc_cols = ["significantWeatherCode", "uvIndex"]
    cont_cols = list(set(frame.columns) - set(disc_cols))
    
    frame[cont_cols] = frame[cont_cols].interpolate()
    frame[disc_cols] = frame[disc_cols].ffill()
    frame = frame.reset_index()
    frame.time = frame.time.map(lambda dt: datetime.strftime(dt, config.DATETIME_FORMAT))
    return frame


def cut_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cut a dataframe on the DateTime column into intervals of 23:00 - 23:00.

    Cuts a given Pandas DataFrame via a string DateTime column into an interval
    of 23:00 - 23:00.

    Args:
        frame (pd.DataFrame): Pandas dataframe to be cut. Must contain a DataTime
            column.

    Returns: pd.DataFrame: Pandas dataframe cut into intervals of 23:00 - 23:00
    """
    datetime_format = config.DATETIME_FORMAT
    hrs = []
    for i in range(len(frame["time"])):
        hrs.append(datetime.strptime(frame["time"][i], datetime_format).hour)
    idxs = np.where(np.asarray(hrs) == 23)[0]
    start = idxs[0]
    end = idxs[2]
    return frame.truncate(start, end)
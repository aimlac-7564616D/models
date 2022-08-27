from datetime import datetime, timedelta
from flask import g
import pandas as pd


def hour_rounder(t: pd.Timestamp) -> pd.Timestamp:
    # Rounds to nearest 0.5 hour
    tNew = t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
    quadrant = t.minute // 15
    if quadrant in [1, 2]:
        tNew += timedelta(hours=0.5)
    elif quadrant == 3:
        tNew += timedelta(hours=1)
    return tNew


def parse_data() -> dict:
    """query the database for data to feed into the co2 saved function"""
    kwargs = {
        "start_date": datetime.utcnow() - timedelta(minutes=60),
        "end_date": datetime.utcnow(),
    }
    data = {
        "co2": 'SELECT time, intensity FROM carbon_dioxide WHERE time > "{start_date}" AND time < "{end_date}"',
        "power": 'SELECT time, wind1, wind2, wind3, wind4, windA, windB, solar, hq_power, computing_center FROM energy_onsite WHERE time > "{start_date}" AND time < "{end_date}"',
    }
    parsed_data = {}
    conn = g.get_conn()

    for key, query in data.items():
        parsed_data[key] = pd.read_sql(query.format(**kwargs), conn)
    return parsed_data


def co2_saved() -> pd.DataFrame:
    data = parse_data()
    # convert power data to nearest 30mins to align with CO2 data
    for i in range(len(data["power"])):
        t = hour_rounder(data["power"].time[i])
        data["power"].time[i] = t

    data = pd.merge(data["co2"], data["power"], on="time", how="inner")
    if len(data):
        data = data.iloc[-1, :]

    netPower = (
        data.wind1
        + data.wind2
        + data.wind3
        + data.wind4
        + data.windA
        + data.windB
        + data.solar
    )

    co2 = pd.DataFrame(
        {
            "time": [data["time"]],
            # 0.5 for 30min sampling to kWh, 1000 to kW from W, 1000 to kg
            "co2saved": [netPower * 0.5 / 1000 * data["intensity"] / 1000],
        }
    )
    return co2

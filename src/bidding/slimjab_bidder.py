import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.bidding.util import get_output_template, register_bidder


@register_bidder(
    "slimjab-bidder",
    args={
        "start_date": date.today() - timedelta(days=1),
        "end_date": date.today() + timedelta(days=2),
    },
    data={
        "power": 'SELECT time, (WindPower + SolarPower - HQPowerDemand) * 1000 AS NetPower FROM powerPrediction WHERE time > "{start_date}" AND time < "{end_date}"'
    },
    default=True,
)
def slimjab_bidder(**kwargs):
    power = kwargs["power"].set_index("time")
    df = get_output_template()
    for i in range(len(df)):
        time = np.datetime64(
            f'{df.loc[i, "applying_date"]} {str(df.loc[i, "hour_ID"] - 1).zfill(2)}'
        )
        if not f"{time}:00:00" in power.index or not f"{time}:30:00" in power.index:
            df = df.drop(i)
            continue

        volume = (
            power.loc[f"{time}:00:00", "NetPower"]
            + power.loc[f"{time}:30:00", "NetPower"]
        ) / 2
        df.loc[i, "volume"] = abs(volume)
        df.loc[i, "price"] = (
            random.random() * 100 + 100
        )  # random values between 100 - 200
        df.loc[i, "type"] = "BUY" if volume < 0 else "SELL"
    return df

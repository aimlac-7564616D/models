from datetime import timedelta
import numpy as np
import pandas as pd

# load weights
with np.load("src/pricing/price_model_weights.npz") as weights:
    w = weights["w"]
    b = weights["b"]


def predict_price_tomorrow(price_df):
    global w, b
    price_df.date = pd.to_datetime(price_df.date, format="%Y-%m-%d %H:%M:%S")
    price_df["datetime"] = price_df.date + price_df.period.map(
        lambda t: timedelta(hours=t - 1)
    )
    price_df = price_df.set_index("datetime").asfreq("H").ffill()
    price_tmr = price_df[["price"]].copy()
    price_tmr.index = price_tmr.index + timedelta(days=1)
    price_tmr.price = np.round(price_df.price.values @ w + b, 2)
    price_tmr = price_tmr.reset_index()
    price_tmr.columns = ["time", "price"]
    return price_tmr

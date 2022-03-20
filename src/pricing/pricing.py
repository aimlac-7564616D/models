# -*- coding: utf-8 -*-
"""Functions to collect energy market data from Elexon.

@author: jadot-bp
"""

import datetime as dt
from typing import List, Tuple

import pandas as pd

# need this to unpack the pickles
import sklearn

import pickle

m_model_p = None
# Load autobidder ML model
with open("src/pricing/model.p", "rb") as handle:
    m_model_p = pickle.load(handle)

m_scalar_p = None
# Load data scaler for autobidder
with open("src/pricing/scaler.p", "rb") as handle:
    m_scalar_p = pickle.load(handle)


def prepare_data_frame(
    b0620_data: pd.DataFrame, b1620_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Adapted from `get_forecast(date, period)` in the slimjab repository

    I think the data and period for both the frames has to be identical for the model later
    down the line to work.

    Returns:
        forecast (pd.DataFrame) : pd.DataFrame containing market forecast
                                  according to model pattern.
    """
    output = pd.merge(b1620_data, b0620_data, on=["settlementDate", "settlementPeriod"])

    output.loc[:, "settlementDate"] = pd.to_datetime(output["settlementDate"])
    output.loc[:, "settlementDate"] -= dt.datetime(1970, 1, 1)  # Since epoch
    output.loc[:, "settlementDate"] = output.loc[:, "settlementDate"].dt.days

    # Pattern dictates current feature arrangement for autobidder model

    pattern = [
        "settlementDate",
        "settlementPeriod",
        "biomass",
        "hydroPumpedStorage",
        "hydroRunofriverAndPoundage",
        "fossilHardCoal",
        "fossilGas",
        "fossilOil",
        "nuclear",
        "other",
        "quantity",
        "solar",
        "windOffshore",
        "windOnshore",
    ]

    return output[pattern]


def get_price_estimate(forecast) -> Tuple[str, int, float]:
    """Calculates the estimated market sell price for the given date and
       period.

    Returns:
        (str, int, float) tuple representing the forecast date, market period
        and predicted market price.
    """
    global m_scalar_p, m_model_p

    # apparently:
    # > Maximum time delta supported by ML model
    # so we'll need a check to see if the time-difference
    # of the models is
    #
    #   delta = (date - dt.datetime.now()).days * 48 + period
    #   delta > 2 * 48
    #
    # so a 2 day period?
    #
    #  if (delta > 2*48):
    #      price = 'NaN'
    #  else:

    rescaled = m_scalar_p.transform(forecast.values[:])
    price = m_model_p.predict(rescaled)[0]
    return price


def get_price_forecast(forecast) -> list:
    """Generates the Day-Ahead (11pm-11pm) market price forecast.

    Returns:
        np.ndarray : Array of market prices.
    """

    dates = set(forecast["settlementDate"])

    outputs = {}
    for date in dates:

        prices = []

        for i in range(1, 49):
            selection = forecast.loc[
                (forecast["settlementPeriod"] == i)
                & (forecast["settlementDate"] == date)
            ]
            if not selection.empty:
                estimate = get_price_estimate(selection)
                prices.append(estimate)
            else:
                prices.append("NaN")

        outputs[date] = prices

    return pd.DataFrame(outputs)

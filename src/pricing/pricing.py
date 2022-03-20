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


def _response_to_df(response_string: str) -> pd.DataFrame:
    """Converts utf-8 decoded response string to a pandas.DataFrame object.

    Args:
        response_string (str) : utf-8 decoded string from response content

    Returns:
        pandas.DataFrame object
    """
    assert len(response_string) != 0

    # Unpack csv formatted response_string
    data_string = response_string.split("\n")
    header = data_string[4].split(",")
    header[0] = header[0].lstrip("*")  # Catch leading asterisk
    content = [x.split(",") for x in data_string[5:-1]]

    return pd.DataFrame(content, columns=header)


def get_bmrs_report(code: str, date: str, period="*") -> pd.DataFrame:
    """Fetches the specified BMRS report from Elexon.

    Fetches the specified BMRS report from Elexon. and returns a
    pandas.DataFrame object containing the report results.

    Args:
        code (str)          : BMRS report identifier (e.g. "B1440")
        date (str)          : Settlement date
        period (str, int)   : settlementperiod (optional, default is `*`)

    Returns:
        pandas.DataFrame object

    Raises:
        ValueError  : if period not between 1-50 inclusive
        ValueError  : if code not in the list of supported BMRS codes
    """

    APIKEY = get_elexon_key()

    if period == "*":
        pass
    elif int(period) < 1 or int(period) > 50:
        raise ValueError(
            "period must be a str or int representing a number"
            " between 1-48 (inclusive) for a 24 hour day. Default"
            " period is `*` for all periods."
        )

    if code not in CODE_DESCRIPTORS.keys():
        raise ValueError(
            f"code {code} either an invalid BMRS code or is not" " yet supported"
        )

    assert isinstance(code, str)
    assert isinstance(date, str)

    url = (
        f"https://api.bmreports.com/BMRS/{code}/v1?APIKey={APIKEY}"
        f"&SettlementDate={date}&Period={period}&ServiceType=csv"
    )

    # Attempt response from cache
    response = cache_get_hashed(url)
    if response is None or not isinstance(response, Response):
        # No result cached, use API
        response = requests.get(url)
        cache_save_hashed(url, response)

    assert response.status_code == 200

    return _response_to_df(response.content.decode("utf-8"))


def get_bmrs_series(code: str, from_date_str: str, to_date_str: str) -> pd.DataFrame:
    """Collates a series of Day-Ahead BMRS reports into a single dataframe.

    Collates a series of Day-Ahead BMRS reports into a single pandas.DataFrame
    containing the BMRS reporting between the specified dates (inclusive).

    Args:
        code (str)          : BMRS report identifier (e.g. "B1440")
        from_date_str (str)     : Initial settlement date ("YYYY-MM-DD")
        to_date_str (str)       : Final settlement date ("YYYY-MM-DD")

    Returns:
        pandas.DataFrame object

    Raises:
        ValueError  : if end_date is before start_date
        ValueError  : if code not in the list of supported BMRS codes
    """
    assert isinstance(code, str)
    assert isinstance(from_date_str, str)
    assert isinstance(to_date_str, str)

    # Recast dates as datetime.datetime objects
    from_date = dt.datetime.strptime(from_date_str, "%Y-%m-%d")
    to_date = dt.datetime.strptime(to_date_str, "%Y-%m-%d")

    if from_date > to_date:
        raise ValueError("from_date must be before to_date.")

    if code not in CODE_DESCRIPTORS.keys():
        raise ValueError(
            f"code {code} either an invalid BMRS code or is not" " yet supported"
        )

    series = pd.DataFrame()  # Initialise empty data frame for series

    loop_date = from_date  # loop_date counter to iterate over dates

    while loop_date <= to_date:
        date_string = dt.datetime.strftime(loop_date, "%Y-%m-%d")
        try:
            data = get_bmrs_report(code, date_string, period="*")
        except IndexError:  # Catch responses with no data
            data = pd.DataFrame([date_string], columns=["Settlement Date"])
        series = pd.concat([series, data])
        loop_date += dt.timedelta(days=1)

    return series


def get_dersys_data(from_date: str, to_date: str, period=None) -> pd.DataFrame:
    """Fetch derived system wide data.

    Fetches derived system wide data for the dates specified and returns a
    pandas.DataFrame object containing the system report.

    Args:
        from_date (str)     : Initial settlement date ("YYYY-MM-DD")
        to_date (str)       : Final settlement date ("YYYY-MM-DD")
        period (str, int)   : settlementperiod (optional, default is None)

    Returns:
        pandas.DataFrame object
    """

    APIKEY = get_elexon_key()

    assert isinstance(from_date, str)
    assert isinstance(to_date, str)
    assert type(period) in [str, int, type(None)]

    pstring = ""

    if period is not None:
        pstring = f"&Period={period}"  # Catch optional period parameter

    url = (
        f"https://api.bmreports.com/BMRS/DERSYSDATA/v1?APIKey={APIKEY}"
        f"&FromSettlementDate={from_date}&ToSettlementDate={to_date}"
        f"{pstring}&ServiceType=csv"
    )

    header = [
        "Record",
        "Settlement Date",
        "settlementPeriod",
        "SSP (£/MWh)",
        "SBP (£/MWh)",
        "BSAD",
        "PDC",
        "RSP",
        "NIV",
        "RP (£/MWh)",
        "RPRV (MWh)",
        "TSAOV",
        "TSABV",
        "TSTAOV",
        "TSTABV",
        "STPAOV",
        "STPABV",
        "TSASV",
        "TSABV",
        "TSTAdSV",
        "TSTAdBV",
    ]

    # Attempt response from cache
    response = cache_get_hashed(url)
    if response is None or not isinstance(response, Response):
        # No result cached, use API
        response = requests.get(url)
        cache_save_hashed(url, response)

    assert response.status_code == 200

    raw_string = response.content.decode("utf-8")
    data_string = raw_string.split("\n")
    content = [x.split(",") for x in data_string[1:-1]]

    return pd.DataFrame(content, columns=header)


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

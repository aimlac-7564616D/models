import datetime as dt

CODE_DESCRIPTORS = {
    "B1720": "Amount of balancing reserves under contract",
    "B1730": "Prices of procured balancing reserves",
    "B1740": "Accepted aggregated offers",
    "B1750": "Activated balancing energy",
    "B1760": "Prices of activated balancing energy",
    "B1770": "Imbalance prices",
    "B1780": "Aggregated imbalance volumes",
    "B1810": ("CrossBorder balancing volumes of exchanged bids" " and offers"),
    "B1820": "CrossBorder balancing prices",
    "B1830": "CrossBorder balancing energy activated",
    "B0610": "Actual total load per bidding zone",
    "B0620": "Day ahead total load forecast per bidding zone",
    "B1430": "Day ahead aggregated generation",
    "B1440": "Generation forecasts for wind and solar",
    "B1610": "Actual generation output per generation unit",
    "B1620": "Actual aggregated generation perType",
    "B1630": ("Actual or estimated wind and solar power" " generation"),
    "B1320": "Congestion management measure counter-trading",
}


def _to_period(timestamp: dt.datetime) -> int:
    """Computes the market period from a given timestamp.

    Args:
        timestamp (dt.datetime) : Timestamp from which the associated market
                                  period is calculated.

    Returns:
        period (int) : Period corresponding to given timestamp.
    """

    assert isinstance(timestamp, (dt.datetime))

    minute_time = timestamp.hour * 60 + timestamp.minute

    period = minute_time // 30 + 1

    return period


def _to_time(period: int) -> str:
    """Computes timestamp from a given market period.

    Args:
        period (int) : Market period

    Returns:
        timestamp (str) : Timestamp in HH:MM corresponding to the market period
                          (to the nearest 30 minutes).
    """

    assert isinstance(period, int)

    hour = (period - 1) * 30 // 60
    minute = (period - 1) * 30 % 60

    return f"{hour:02}:{minute:02}"

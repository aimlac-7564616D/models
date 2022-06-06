from datetime import date, timedelta
from src.bidding.util import register_bidder, get_output_template
import pandas as pd
import random


@register_bidder(
    "bogo-bidder",
    args={
        "start_date": date.today() - timedelta(days=1),
        "end_date": date.today() + timedelta(days=1),
    },
    data={
        # input will be processed into a pd.DataFrame can be accessed in the function
        "energy": 'SELECT time FROM energy_onsite WHERE time > "{start_date}" AND time < "{end_date}"'
    },
)
def bogo_bidder(**kwargs):
    """Example of bidder function."""

    # access args
    start_date = kwargs["start_date"]
    end_date = kwargs["end_date"]

    # access data
    energy = kwargs["energy"]

    # get output template, the orders must be 2 days ahead to be accepted
    df = get_output_template()
    for i in range(len(df)):
        df.loc[i, "volume"] = (
            random.random() * 100 + 50
        )  # random values between 50 - 150
        df.loc[i, "price"] = (
            random.random() * 100 + 50
        )  # random values between 50 - 150
        df.loc[i, "type"] = random.choice(["BUY", "SELL"])
    return df

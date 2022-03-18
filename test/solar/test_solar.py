from src.solar import get_solar_prediction

import numpy as np


def test_get_solar_prediction(timeseries):
    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.99137390910548,
            278.47015128705357,
            431.9787773779482,
            469.0000000000001,
            469.0000000000001,
            469.0000000000001,
            469.0000000000001,
            469.0000000000001,
            469.0000000000001,
            460.54216284454753,
            390.15362216922955,
            203.32304185112525,
            39.21158252644325,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    result = get_solar_prediction(timeseries)
    assert all(result["SolarPower"] == expected)

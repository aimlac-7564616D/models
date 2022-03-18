from src.demand import get_energy_demand

import numpy as np


def test_get_energy_demand(timeseries):
    expected_hq_temperature = np.array(
        [
            6.85,
            6.665,
            6.48,
            6.32,
            6.16,
            5.95,
            5.74,
            5.49,
            5.24,
            5.045,
            4.85,
            4.744999999999999,
            4.64,
            4.585,
            4.53,
            4.715,
            4.9,
            5.385,
            5.87,
            6.5649999999999995,
            7.26,
            8.075,
            8.89,
            9.695,
            10.5,
            11.115,
            11.73,
            12.055,
            12.38,
            12.525,
            12.67,
            12.695,
            12.72,
            12.36,
            12.0,
            11.440000000000001,
            10.88,
            10.155000000000001,
            9.43,
            8.82,
            8.21,
            7.700000000000001,
            7.19,
            6.710000000000001,
            6.23,
            5.925000000000001,
            5.62,
            5.32
        ]
    )
    expected_total_demand = np.array(
        [
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0
        ]
    )

    result = get_energy_demand(timeseries)
    assert all(np.isclose(result['HQ Temperature'], expected_hq_temperature))
    assert all(np.isclose(result['Total demand'], expected_total_demand))

# copied from the SLIMJaB repository
"""Tests for the building energy demand functions.

Todo:
    test_get_office_equipment_demand() Should account for edge cases.
    test_get_lighting_and_other_demand() Should account for edge cases.

"""

import pytest

import pandas as pd
import numpy as np

from src.onsite.onsite import (
    get_data_centre_demand,
    get_energy_demand,
    get_office_equipment_demand,
    get_lighting_and_other_demand,
)


def test_get_data_centre_demand() -> None:
    """Test to ensure data center demand is within reasonable ranges."""
    output = get_data_centre_demand()

    # Check return type
    assert isinstance(output, np.ndarray)

    # Check return shape
    assert output.shape == (48,)

    assert sum(output) <= 9600
    assert sum(output) >= 0


def test_get_energy_demand(timeseries) -> None:
    """Test to get the energy demand of the entire building."""
    output = get_energy_demand(timeseries)

    assert isinstance(output, pd.DataFrame)

    # assert number of records
    assert output.shape[0] == 48

    # assert columns
    # assert output.shape[1] == 7
    assert output.shape[1] == 8  # added a column: HQ Temperature

    assert "Active office mask" in output
    assert "Heating" in output
    assert "Data Centre" in output
    assert "Office Equipment" in output
    assert "LightingOther" in output
    assert "Total demand" in output
    assert "HQ Temperature" in output

    assert isinstance(output["Active office mask"][0], np.bool_)
    assert isinstance(output["Heating"][0], float)
    assert isinstance(output["Data Centre"][0], float)
    assert isinstance(output["LightingOther"][0], float)
    assert isinstance(output["Total demand"][0], float)

    # Ensure total column is correct
    assert (
        output["Total demand"]
        == output["Heating"]
        + output["Data Centre"]
        + output["Office Equipment"]
        + output["LightingOther"]
    ).all()


@pytest.mark.parametrize(
    "active_office_mask",
    (
        [
            False,
        ]
        * 20
        + [
            True,
        ]
        * 16
        + [
            False,
        ]
        * 12,
    ),
)
def test_get_office_equipment_demand(active_office_mask):
    """Test to ensure office cumulative office demand is the expected result."""
    output = get_office_equipment_demand(active_office_mask)

    # Check return type
    assert isinstance(output, np.ndarray)

    # Check return shape
    assert output.shape == (48,)

    assert sum(output) == 160


@pytest.mark.parametrize(
    "active_office_mask",
    (
        [
            False,
        ]
        * 20
        + [
            True,
        ]
        * 16
        + [
            False,
        ]
        * 12,
    ),
)
# @pytest.mark.skip(reason="TypeError: get_office_equipment_demand() missing 1 required positional argument: 'active_office_mask'")
def test_get_lighting_and_other_demand(active_office_mask):
    """Test to ensure lighting and misc demand is the expected result."""
    output = get_lighting_and_other_demand(active_office_mask)

    # Check return type
    assert isinstance(output, np.ndarray)

    # Check return shape
    assert output.shape == (48,)

    assert sum(output) == 320

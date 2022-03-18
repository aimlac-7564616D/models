import json

import pandas
import pytest

from test.samples.sample_data import sample_time_series


@pytest.fixture(scope="session")
def timeseries() -> pandas.DataFrame:
    yield pandas.read_json(json.dumps(sample_time_series))

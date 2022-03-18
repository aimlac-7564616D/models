from src.onsite.onsite import get_energy_demand
from src.solar.solar import get_solar_prediction
from src.wind.wind import get_wind_prediction

from flask import Blueprint, request
from pydash.objects import get

import json
import pandas as pd

bp = Blueprint("power", __name__, url_prefix="/power")


@bp.route("/predict-solar", methods=["POST"])
def predict_solar():
    try:
        forecast_json = get(request.json, "features[0].properties.timeSeries")
        forecast_df = pd.read_json(json.dumps(forecast_json))
        solar_output_df = get_solar_prediction(forecast_df)
        solar_output_json = solar_output_df.to_json(orient="records")
        return solar_output_json
    except Exception as e:
        return {"message": str(e)}, 500


@bp.route("/predict-wind", methods=["POST"])
def predict_wind():
    try:
        forecast_json = get(request.json, "features[0].properties.timeSeries")
        forecast_df = pd.read_json(json.dumps(forecast_json))
        wind_report_df = get_wind_prediction(forecast_df)
        wind_report_json = wind_report_df.to_json(orient="records")
        return wind_report_json
    except Exception as e:
        return {"message": str(e)}, 500


@bp.route("/predict-onsite", methods=["POST"])
def predict_onsite():
    try:
        forecast_json = get(request.json, "features[0].properties.timeSeries")
        forecast_df = pd.read_json(json.dumps(forecast_json))
        demand_df = get_energy_demand(forecast_df)
        demand_df.rename(columns={
            "DateTime": "time",
            "HQ Temperature": "HQTemperature",
            "Total demand": "HQPowerDemand"
        }, inplace=True)
        demand_df = demand_df[["time", "HQPowerDemand", "HQTemperature"]]
        demand_json = demand_df.to_json(orient="records")
        return demand_json
    except Exception as e:
        return {"message": str(e)}, 500

import json

import pandas as pd
from flask import Blueprint

from src.pricing import predict_price_tomorrow

bp = Blueprint("price", __name__, url_prefix="/price")


@bp.route("/predict-price", methods=["GET"])
def predict_price():
    try:
        prices_df = predict_price_tomorrow()
        prices_df = prices_df.reset_index()
        prices_df.columns = ["time", "price"]
        prices_json = prices_df.to_json(orient="records")
        return prices_json
    except Exception as e:
        return {"message": str(e)}, 500

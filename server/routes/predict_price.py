import json

import pandas as pd
from flask import Blueprint, request

from src.pricing import predict_price_tomorrow

bp = Blueprint("price", __name__, url_prefix="/price")


@bp.route("/predict-price", methods=["POST"])
def predict_price():
    try:
        price_df = pd.read_json(json.dumps(request.json))
        price_tmr = predict_price_tomorrow(price_df)
        price_json = price_tmr.to_json(orient="records")
        return price_json
    except Exception as e:
        return {"message": str(e)}, 500

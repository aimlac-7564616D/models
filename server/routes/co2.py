from flask import Blueprint
from src.co2.co2_saved import co2_saved

bp = Blueprint("co2", __name__, url_prefix="/co2")

@bp.route("/set", methods=["GET"])
def setCO2Saved():
    try:
        co2 = co2_saved()
        co2_json = co2.to_json(orient="records")
        return co2_json
    except Exception as e:
        return {"message": str(e)}, 500

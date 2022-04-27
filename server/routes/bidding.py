from flask import Blueprint, g, request
from datetime import date, timedelta
from src.bidding import BIDDERS

bp = Blueprint("bid", __name__, url_prefix="/bid")


@bp.route("/list", methods=["GET"])
def list_bidders():
    try:
        keys = list(BIDDERS.keys())
        if "default" in keys:
            keys.remove("default")
        default = "none"
        for key in keys:
            if BIDDERS[key] == BIDDERS.get("default"):
                default = key
        mssg = {"bidders": keys, "default": default}
        return mssg, 200
    except Exception as e:
        return {"message": str(e)}, 500


@bp.route("/set", methods=["POST"])
def set_bids():
    try:
        if request.data and request.json:
            bidder = request.json.get("bidder", "default")
        else:
            bidder = "default"
        resp = BIDDERS[bidder]()
        return resp.json()
    except Exception as e:
        return {"message": str(e)}, 500

import server.routes.bidding as bidding
import server.routes.predict_power as predict_power

# create binding to export
routes = [bidding.bp, predict_power.bp]

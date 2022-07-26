import server.routes.bidding as bidding
import server.routes.co2 as co2
import server.routes.predict_power as predict_power
import server.routes.predict_price as predict_price


# create binding to export
routes = [bidding.bp, predict_power.bp, co2.bp, predict_price.bp]

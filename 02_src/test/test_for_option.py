from classes.options.Vanilla import VanillaCall
from datetime import date
vc = VanillaCall()
vc.set_paras(notional=12e6,start_date=date(2021,7,21),end_date=date(2021,9,22),K=5.42,option_fee=1780800,
             stock_code='000959.SZ',start_price=6.02)
vc.calculate_greeks()
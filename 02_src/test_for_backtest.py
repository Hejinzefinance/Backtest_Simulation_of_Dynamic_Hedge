from classes.backtest.backtestFramework import BacktestFramework
from datetime import date
btf = BacktestFramework()
btf.add_option('VanillaCall',-1,notional=12e6,start_date=date(2021,7,21),end_date=date(2021,9,22),K=5.42,option_fee=1780800,
             stock_code='000959.SZ',start_price=6.02)
btf.set_strategy('HedgeHalf')
btf.run_backtest()
btf.visualize()
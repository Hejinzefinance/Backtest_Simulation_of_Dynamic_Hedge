from classes.backtest.backtestFramework import BacktestFramework
from datetime import date
#%% 香草期权
btf = BacktestFramework()
btf.set_option('VanillaCall',-1,notional=12e6,start_date=date(2021,7,21),end_date=date(2021,9,22),K=5.42,option_fee=1780800,
              stock_code='000959.SZ',start_price=6.02)
btf.set_strategy('HedgeAll')
btf.run_backtest()
btf.visualize(report=True)
btf.set_strategy('HedgeHalf')
btf.run_backtest()
btf.visualize(report=True)
#%% 参与型看涨
btf = BacktestFramework()
btf.set_option('ParticipateCall',-1,notional=5e6,start_date=date(2021,6,15),end_date=date(2021,7,15),K_low=3.43,
                K_high=3.81,option_fee=555000,p_ratio1=1,p_ratio2=0.9,stock_code='600808.SH',start_price=3.81)
btf.set_strategy('HedgeAll')
btf.run_backtest()
btf.visualize(report=True)
btf.set_strategy('HedgeHalf')
btf.run_backtest()
btf.visualize(report=True)
#%% 看涨价差
btf = BacktestFramework()
btf.set_option('SpreadCall',-1,notional=2e7,start_date=date(2021,6,15),end_date=date(2021,7,15),
               K_low=3.68,K_high=5.79,option_fee=5776000,stock_code='300180.SZ',start_price=5.26)
btf.set_strategy('HedgeAll')
btf.run_backtest()
btf.visualize(report=True)
btf.set_strategy('HedgeHalf')
btf.run_backtest()
btf.visualize(report=True)
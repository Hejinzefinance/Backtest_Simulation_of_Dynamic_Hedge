import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..options.OptionPortfolio import OptionPortfolio
from ..basicData.basicData import BasicData
from ..strategy import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BacktestFramework:
    ALL_TRADE_DATES = BasicData.ALL_TRADE_DATES
    TR = 0.0002
    def __init__(self):
        self.backtest_columns = ['option_value','stock_position','stock_value','trading_cost','option_pnl',
                                 'stock_pnl','delta_nav','nav','cash_account','delta_pnl','gamma_pnl','vega_pnl',
                                 'theta_pnl','high_order_pnl','unhedged_pnl','total_nav','trade_dummy']
        self.analysis_columns = ['unhedge_delta_value','gamma_value','vega_value','theta_value','high_order_value',
                                 'cum_trading_cost','total_nav']
        self.analysis_index = ['total_pnl','option_pnl','stock_pnl','trading_cost','min_cash','max_drawdown',
                               'total_unhedged_pnl','total_gamma_pnl','total_vega_pnl','total_theta_pnl',
                               'total_high_order_pnl']
        self.reset()

    def reset(self):
        self.option_portfolio = OptionPortfolio()
        self.set_strategy()

    def set_strategy(self, strategy_name=''):
        self.strategy_name = strategy_name
        self.strategy = None if not strategy_name else eval(strategy_name)()

    def add_option(self,option_class,option_position,**option_paras):
        self.option_portfolio.add_option_by_dict(option_class,option_position,option_paras)
        
    
    def run_backtest(self):
        self.init_backtest()
        self.backtest_df.loc[:, 'stock_position'] = self.strategy.get_hedging_position(self.option_portfolio.get_greeks(),self.stock_prices)
        self.backtest_df.loc[:, 'stock_value'] = self.backtest_df.loc[:, 'stock_position']*self.stock_prices
        augmented_stock_position = np.hstack((0,self.backtest_df.loc[:, 'stock_position']))
        self.backtest_df.loc[:, 'trading_cost'] = np.abs(np.diff(augmented_stock_position))*self.stock_prices*self.TR
        self.backtest_df.loc[:, 'option_pnl'] = self.backtest_df.loc[:, 'option_value'].diff().fillna(0)
        self.backtest_df.loc[:, 'stock_pnl'] = (self.backtest_df.loc[:, 'stock_position'].shift(1)*self.stock_prices.diff()).fillna(0)
        self.backtest_df.loc[:, 'delta_nav'] = self.backtest_df.loc[:, 'stock_pnl']-self.backtest_df.loc[:, 'trading_cost']
        self.backtest_df.loc[:, 'nav'] = np.cumsum(self.backtest_df.loc[:, 'delta_nav'])
        self.backtest_df.loc[:, 'cash_account'] = self.option_portfolio.option_fee+self.backtest_df.loc[:, 'nav']-self.backtest_df.loc[:, 'stock_value']
        pct_change = self.stock_prices.pct_change().fillna(0)
        self.backtest_df.loc[:, 'delta_pnl'] = (self.option_portfolio.greek_df.loc[:,'cash_delta'].shift(1)*pct_change).fillna(0)
        self.backtest_df.loc[:, 'gamma_pnl'] = (0.5*self.option_portfolio.greek_df.loc[:,'cash_gamma'].shift(1)*100*np.power(pct_change,2)).fillna(0)
        self.backtest_df.loc[:, 'vega_pnl'] = (self.option_portfolio.greek_df.loc[:,'pos_vega'].shift(1)*self.vol.diff(1)).fillna(0)
        self.backtest_df.loc[:, 'theta_pnl'] = self.option_portfolio.greek_df.loc[:,'cash_theta'].shift(1).fillna(0)
        self.backtest_df.loc[:, 'high_order_pnl'] = self.backtest_df.loc[:, 'option_pnl']-self.backtest_df.loc[:, 'delta_pnl']-self.backtest_df.loc[:, 'gamma_pnl']-self.backtest_df.loc[:, 'vega_pnl']-self.backtest_df.loc[:, 'theta_pnl']
        self.backtest_df.loc[:, 'unhedged_pnl'] = self.backtest_df.loc[:, 'stock_pnl']+self.backtest_df.loc[:, 'delta_pnl']
        self.backtest_df.loc[:, 'total_nav'] = self.backtest_df.loc[:, 'cash_account']+self.backtest_df.loc[:, 'option_value']+self.backtest_df.loc[:, 'stock_value']
        self.backtest_df.loc[self.backtest_df.loc[:, 'stock_position'].diff()==0, 'trade_dummy'] = 0
        self.backtest_df.loc[:, 'trade_dummy'].fillna(1,inplace=True)
        self.trade_ticks = self.backtest_df.loc[self.backtest_df.loc[:, 'trade_dummy']==1].index
        self.analysis()

    def analysis(self):
        self.init_analysis()
        self.analysis_df.loc[:,'unhedge_delta_value'] = self.backtest_df.loc[:,'unhedged_pnl'].cumsum()
        self.analysis_df.loc[:, 'gamma_value'] = self.backtest_df.loc[:, 'gamma_pnl'].cumsum()
        self.analysis_df.loc[:, 'vega_value'] = self.backtest_df.loc[:, 'vega_pnl'].cumsum()
        self.analysis_df.loc[:, 'theta_value'] = self.backtest_df.loc[:, 'theta_pnl'].cumsum()
        self.analysis_df.loc[:, 'high_order_value'] = self.backtest_df.loc[:, 'high_order_pnl'].cumsum()
        self.analysis_df.loc[:, 'cum_trading_cost'] = self.backtest_df.loc[:, 'trading_cost'].cumsum()
        self.analysis_df.loc[:, 'total_value'] = self.backtest_df['total_nav']-self.backtest_df['total_nav'].iloc[0]
        self.analysis_dict['total_pnl'] = self.backtest_df['total_nav'].iloc[-1]-self.backtest_df['total_nav'].iloc[0]
        self.analysis_dict['option_pnl'] = self.backtest_df['option_value'].iloc[-1]-self.backtest_df['option_value'].iloc[0]
        self.analysis_dict['stock_pnl'] = self.backtest_df['stock_pnl'].sum()
        self.analysis_dict['trading_cost'] = self.backtest_df['trading_cost'].sum()
        self.analysis_dict['min_cash'] = self.backtest_df['cash_account'].min()
        self.analysis_dict['max_drawdown'] = self.cal_MDD(self.backtest_df.loc[:, 'total_nav'])
        self.analysis_dict['total_unhedged_pnl'] = self.analysis_df['unhedge_delta_value'].iloc[-1]
        self.analysis_dict['total_gamma_pnl'] = self.analysis_df['gamma_value'].iloc[-1]
        self.analysis_dict['total_vega_pnl'] = self.analysis_df['vega_value'].iloc[-1]
        self.analysis_dict['total_theta_pnl'] = self.analysis_df['theta_value'].iloc[-1]
        self.analysis_dict['total_high_order_pnl'] = self.analysis_df['high_order_value'].iloc[-1]

    def visualize(self):
        fig1,ax1 = self.init_canvas([0.05,0.05,0.9,0.84])
        ax1.plot(self.trade_datetimes,self.backtest_df['stock_value']+self.backtest_df['cash_account'],
                 linewidth=0.5,color='blue',label='hedge_value')
        ax1.plot(self.trade_datetimes,self.backtest_df['option_value'],
                 linewidth=0.5,color='orange',label='option_value')
        ax1.plot(self.trade_datetimes,self.backtest_df['total_nav'],
                 linewidth=0.5,color='red',label='total_value')
        t_data = self.backtest_df.loc[:,['stock_value','option_value','total_nav']]
        min_value = t_data.min().min()
        max_value = t_data.max().max()
        ax1.vlines(self.trade_ticks,min_value,max_value,label='hedge_tradings',
                   color='black',linewidths=0.1)
        ax2 = ax1.twinx()
        ax2.plot(self.trade_datetimes,self.backtest_df['stock_position'],
                 linewidth=0.5,color='green',label='stock_position')
        ax1.set_title('股票期权对冲总览，对冲策略:{11:s}\n期权类型:{0:s}，名义金额:{1:.0f}，标的:{2:s}，期权费:{3:.0f}，执行价:{4:.2f}，整体盈利:{5:.0f}\n期权损益:{6:.0f}，对冲损益:{7:.0f}，交易成本:{8:.0f}，现金最大占用量:{9:.0f}，组合最大回撤:{10:.0f}'.format(\
            self.option_portfolio.option_name,self.option_portfolio.notional,
            self.option_portfolio.stock_code,self.option_portfolio.option_fee,
            self.option_portfolio.strike_price,self.analysis_dict['total_pnl'],
            self.analysis_dict['option_pnl'],self.analysis_dict['stock_pnl'],
            self.analysis_dict['trading_cost'],-self.analysis_dict['min_cash'],
            self.analysis_dict['max_drawdown'],self.strategy_name))
        fig1.legend(loc='right')
        plt.show()
        
        fig2,ax1 = self.init_canvas([0.05,0.05,0.9,0.84])
        ax1.plot(self.trade_datetimes,self.analysis_df['total_value'],
                 linewidth=1,color='red',label='total_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['unhedge_delta_value'],
                 linewidth=0.5,color='pink',label='unhedge_delta_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['gamma_value'],
                 linewidth=0.5,color='green',label='gamma_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['vega_value'],
                 linewidth=0.5,color='orange',label='vega_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['theta_value'],
                 linewidth=0.5,color='purple',label='theta_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['high_order_value'],
                 linewidth=0.5,color='brown',label='high_order_value')
        ax1.plot(self.trade_datetimes,self.analysis_df['cum_trading_cost'],
                 linewidth=0.5,color='black',label='cum_trading_cost')
        t_data = self.analysis_df
        min_value = t_data.min().min()
        max_value = t_data.max().max()
        ax1.vlines(self.trade_ticks,min_value,max_value,label='hedge_tradings',
                   color='black',linewidths=0.1)
        ax1.set_title('股票期权对冲总览，对冲策略:{11:s}\n期权类型:{0:s}，名义金额:{1:.0f}，标的:{2:s}，期权费:{3:.0f}，执行价:{4:.2f}，整体盈利:{5:.0f}\nunhedged_delta损益:{6:.0f}，gamma损益:{7:.0f}，vega损益:{8:.0f}，theta损益:{9:.0f}，高阶损益:{10:.0f}，交易成本:{11:.0f}'.format(\
            self.option_portfolio.option_name,self.option_portfolio.notional,
            self.option_portfolio.stock_code,self.option_portfolio.option_fee,
            self.option_portfolio.strike_price,self.analysis_dict['total_pnl'],
            self.analysis_dict['total_unhedged_pnl'],self.analysis_dict['total_gamma_pnl'],
            self.analysis_dict['total_vega_pnl'],-self.analysis_dict['total_theta_pnl'],
            self.analysis_dict['total_high_order_pnl'],self.analysis_dict['trading_cost'],
            self.strategy_name))
        fig2.legend(loc='right')
        
        
    def init_analysis(self):
        self.analysis_df = pd.DataFrame(index=self.trade_datetimes,columns=self.analysis_columns)
        self.analysis_dict = dict().fromkeys(self.analysis_index)

    @staticmethod
    def init_canvas(rect=[0.05, 0.05, 0.9, 0.9]):
        fig = plt.figure(figsize=(11.694, 8), dpi=300)
        ax = fig.add_axes(rect=rect)
        return fig,ax

    def init_backtest(self):
        self.stock_prices = self.option_portfolio.public_df.loc[:, 'stock_price']
        self.vol = self.option_portfolio.public_df.loc[:,'sigma']
        self.trade_datetimes = self.option_portfolio.trade_datetimes
        self.backtest_df = pd.DataFrame(index=self.trade_datetimes,columns=self.backtest_columns)
        self.backtest_df.loc[:, 'option_value'] = self.option_portfolio.greek_df.loc[:, 'option_value']

    @staticmethod
    def cal_MDD(series):
        return np.max(series.cummax()-series)



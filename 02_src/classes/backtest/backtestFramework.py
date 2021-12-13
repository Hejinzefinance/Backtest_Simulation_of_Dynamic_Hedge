import numpy as np
import pandas as pd
import os
from datetime import datetime
from ..options.OptionPortfolio import OptionPortfolio
from matplotlib.backends.backend_pdf import PdfPages
from ..basicData.basicData import BasicData
from ..strategy import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BacktestFramework:
    ALL_TRADE_DATES = BasicData.ALL_TRADE_DATES
    TR = 0.0002
    def __init__(self):
        self.backtest_columns = ['option_value','sigma','stock_price','stock_position','stock_value','stock_pnl','trading_cost',
                                 'delta_nav', 'nav', 'cash_account','option_pnl','cash_delta','cash_gamma',
                                 'total_cash_delta','delta_pnl','gamma_pnl','other_pnl','delta_exposure_pnl',
                                 'exposure_direction','delta_value','gamma_value','other_value','trading_cum_cost',
                                 'trade_dummy','trade_type','delta_pnl_contribution']
        self.analysis_index = ['total_pnl','option_pnl','stock_pnl','trading_cost','delta_pnl','gamma_pnl',
                               'other_pnl','delta_pnl_part1','delta_pnl_part2','delta_pnl_part3',
                               'delta_pnl_part4','min_cash','max_drawdown']
        self.reset()
        self.CLASS_PATH = __file__
        self.REPORT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'report')

    def reset(self):
        self.option_portfolio = OptionPortfolio()
        self.set_strategy()

    def set_strategy(self, strategy_name=''):
        self.strategy_name = strategy_name
        self.strategy = None if not strategy_name else eval(strategy_name)()

    def set_option(self,option_class,option_position,**option_paras):
        self.option_portfolio.add_option_by_dict(option_class,option_position,option_paras)

    def run_backtest(self):
        self.init_backtest()
        self.backtest_df.loc[:, 'stock_position'] = self.strategy.get_hedging_position(self.option_portfolio.get_greeks(),self.stock_prices)
        self.backtest_df.loc[:, 'stock_value'] = self.backtest_df.loc[:, 'stock_position']*self.stock_prices
        augmented_stock_position = np.hstack((0,self.backtest_df.loc[:, 'stock_position']))
        self.backtest_df.loc[:, 'trading_cost'] = np.abs(np.diff(augmented_stock_position))*self.stock_prices*self.TR
        self.backtest_df.loc[:, 'option_pnl'] = self.backtest_df.loc[:, 'option_value'].diff().fillna(0)
        self.backtest_df.loc[:, 'stock_pnl'] = (self.backtest_df.loc[:, 'stock_position'].shift(1)*self.stock_prices.diff()).fillna(0)
        self.backtest_df.loc[:, 'delta_nav'] = self.backtest_df.loc[:, 'stock_pnl']+self.backtest_df.loc[:, 'option_pnl']-self.backtest_df.loc[:, 'trading_cost']
        self.backtest_df.loc[:, 'nav'] = np.cumsum(self.backtest_df.loc[:, 'delta_nav'])+self.option_portfolio.option_fee+self.backtest_df.loc[:, 'option_value'].iloc[0]
        self.backtest_df.loc[:, 'cash_account'] = self.backtest_df.loc[:, 'nav']-self.backtest_df.loc[:, 'stock_value']-self.backtest_df.loc[:, 'option_value']
        self.backtest_df.loc[:, 'cash_delta'] = self.option_portfolio.greek_df.loc[:,'cash_delta']
        self.backtest_df.loc[:, 'cash_gamma'] = self.option_portfolio.greek_df.loc[:,'cash_gamma']
        self.backtest_df.loc[:, 'total_cash_delta'] = self.backtest_df.loc[:, 'cash_delta']+self.backtest_df.loc[:, 'stock_position']*self.stock_prices
        pct_change = self.stock_prices.pct_change().fillna(0)
        self.backtest_df.loc[:, 'delta_pnl'] = (self.backtest_df.loc[:, 'cash_delta'].shift(1)*pct_change).fillna(0)
        self.backtest_df.loc[:, 'delta_exposure_pnl'] = (self.backtest_df.loc[:, 'total_cash_delta'].shift(1)*pct_change).fillna(0)
        self.backtest_df.loc[:, 'gamma_pnl'] = (0.5*self.option_portfolio.greek_df.loc[:,'cash_gamma'].shift(1)*100*np.power(pct_change,2)).fillna(0)
        self.backtest_df.loc[:, 'other_pnl'] = self.backtest_df.loc[:, 'option_pnl']-self.backtest_df.loc[:, 'delta_pnl']-self.backtest_df.loc[:, 'gamma_pnl']
        self.backtest_df.loc[self.backtest_df.loc[:, 'total_cash_delta']>=0, 'exposure_direction'] = 1
        self.backtest_df.loc[:, 'exposure_direction'].fillna(-1, inplace=True)
        self.backtest_df.loc[:, 'exposure_direction'] = self.backtest_df.loc[:,'exposure_direction'].shift(1)
        self.backtest_df.loc[:, 'exposure_direction'].fillna(0,inplace=True)
        self.backtest_df.loc[:,'delta_value'] = self.backtest_df.loc[:, 'delta_exposure_pnl'].cumsum()
        self.backtest_df.loc[:, 'gamma_value'] = self.backtest_df.loc[:, 'gamma_pnl'].cumsum()
        self.backtest_df.loc[:, 'other_value'] = self.backtest_df.loc[:, 'other_pnl'].cumsum()
        self.backtest_df.loc[:, 'trading_cum_cost'] = self.backtest_df.loc[:, 'trading_cost'].cumsum()
        self.backtest_df.loc[self.backtest_df.loc[:, 'stock_position'].diff()==0, 'trade_dummy'] = 0
        self.backtest_df.loc[:, 'trade_dummy'].fillna(1,inplace=True)
        self.trade_ticks = self.backtest_df.loc[self.backtest_df.loc[:, 'trade_dummy']==1].index
        self.backtest_df.loc[(self.backtest_df.loc[:, 'exposure_direction']>0)&(self.backtest_df.loc[:, 'stock_price'].diff()>0), 'trade_type'] = 1
        self.backtest_df.loc[(self.backtest_df.loc[:, 'exposure_direction']<0)&(self.backtest_df.loc[:, 'stock_price'].diff()>0), 'trade_type'] = 2
        self.backtest_df.loc[(self.backtest_df.loc[:, 'exposure_direction']<0)&(self.backtest_df.loc[:, 'stock_price'].diff()<0), 'trade_type'] = 3
        self.backtest_df.loc[(self.backtest_df.loc[:, 'exposure_direction']>0)&(self.backtest_df.loc[:, 'stock_price'].diff()<0), 'trade_type'] = 4
        self.backtest_df.loc[:, 'trade_type'].fillna(0, inplace=True)
        for trade_type in [1,2,3,4]:
            self.backtest_df.loc[self.backtest_df.trade_type==trade_type,'delta_pnl_contribution'] = self.backtest_df.loc[self.backtest_df.trade_type==trade_type,'delta_exposure_pnl']/self.backtest_df.loc[self.backtest_df.trade_type==trade_type,'delta_exposure_pnl'].sum()
        self.analysis()

    def analysis(self):
        self.init_analysis()
        self.analysis_dict['total_pnl'] = self.backtest_df['nav'].iloc[-1]
        self.analysis_dict['option_pnl'] = self.backtest_df['option_value'].iloc[-1]+self.option_portfolio.option_fee
        self.analysis_dict['stock_pnl'] = self.backtest_df['stock_pnl'].sum()
        self.analysis_dict['trading_cost'] = self.backtest_df['trading_cum_cost'].iloc[-1]
        self.analysis_dict['delta_pnl'] = self.backtest_df.loc[:,'delta_value'].iloc[-1]
        self.analysis_dict['gamma_pnl'] = self.backtest_df['gamma_value'].iloc[-1]
        self.analysis_dict['other_pnl'] = self.backtest_df['other_value'].iloc[-1]
        self.analysis_dict['delta_pnl_part1'] = self.backtest_df.loc[self.backtest_df.trade_type == 1, 'delta_exposure_pnl'].sum()
        self.analysis_dict['delta_pnl_part2'] = self.backtest_df.loc[self.backtest_df.trade_type == 2, 'delta_exposure_pnl'].sum()
        self.analysis_dict['delta_pnl_part3'] = self.backtest_df.loc[self.backtest_df.trade_type == 3, 'delta_exposure_pnl'].sum()
        self.analysis_dict['delta_pnl_part4'] = self.backtest_df.loc[self.backtest_df.trade_type == 4, 'delta_exposure_pnl'].sum()
        self.analysis_dict['min_cash'] = self.backtest_df['cash_account'].min()
        self.analysis_dict['max_drawdown'] = self.cal_MDD(self.backtest_df.loc[:, 'nav'])

    def visualize(self,report=False):
        # 对冲总览
        fig1,ax1 = self.init_canvas([0.08,0.05,0.85,0.81])
        ax1.plot(self.trade_dates,self.backtest_df['stock_value']+self.backtest_df['cash_account'],
                 linewidth=0.5,color='blue',label='hedge_value')
        ax1.plot(self.trade_dates,self.backtest_df['option_value'],
                 linewidth=0.5,color='orange',label='option_value')
        ax1.plot(self.trade_dates,self.backtest_df['nav'],
                 linewidth=0.5,color='red',label='total_value')
        t_data = self.backtest_df.loc[:,['stock_value','option_value','nav']]
        min_value = t_data.min().min()
        max_value = t_data.max().max()
        ax1.vlines(self.trade_ticks,min_value,max_value,label='hedge_tradings',
                   color='black',linewidths=0.1)
        ax2 = ax1.twinx()
        ax2.plot(self.trade_dates,self.backtest_df['stock_position'],
                 linewidth=0.5,color='green',label='stock_position（右轴）')
        ax1.set_title('股票期权对冲总览，对冲策略:{0:s}\n{1:s}\n整体盈利:{2:,.0f}，期权损益:{3:,.0f}，对冲损益:{4:,.0f}，交易成本:{5:,.0f}\n现金最大占用量:{6:,.0f}，组合最大回撤:{7:,.0f}'.format(\
            self.strategy_name,self.option_portfolio.option_info,self.analysis_dict['total_pnl'],
            self.analysis_dict['option_pnl'],self.analysis_dict['stock_pnl'],
            self.analysis_dict['trading_cost'],-self.analysis_dict['min_cash'],
            self.analysis_dict['max_drawdown']))
        fig1.legend(loc='right')
        ax1.set_ylabel('金额/元')
        ax2.set_ylabel('股票头寸/股')
        plt.show()
        # 收益拆分
        fig2,ax1 = self.init_canvas([0.09,0.05,0.88,0.81])
        ax1.plot(self.trade_dates,self.backtest_df['nav'],linewidth=1,
                 color='red',label='整体盈利')
        ax1.plot(self.trade_dates,self.backtest_df['delta_value'],linewidth=0.5,
                 color='orange',label='delta敞口盈利')
        ax1.plot(self.trade_dates,self.backtest_df['gamma_value'],linewidth=0.5,
                 color='green',label='gamma盈利')
        ax1.plot(self.trade_dates,self.backtest_df['other_value'],linewidth=0.5,
                 color='blue',label='其他盈利')
        ax1.plot(self.trade_dates,-self.backtest_df['trading_cum_cost'],linewidth=0.5,
                 color='grey',label='交易成本')
        t_data = self.backtest_df.loc[:,['delta_value','gamma_value','other_value','trading_cum_cost','nav']]
        min_value = t_data.min().min()
        max_value = t_data.max().max()
        ax1.vlines(self.trade_ticks,min_value,max_value,label='hedge_tradings',
                   color='black',linewidths=0.1)
        ax1.legend(loc='best')
        ax1.set_ylabel('金额/元')
        ax1.set_title('对冲收益分解，对冲策略:{0:s}\n{1:s}\n整体盈利:{2:,.0f}，期权初始价值:{7:,.0f}，delta敞口盈利:{3:,.0f}，gamma盈利:{4:,.0f}\n其他盈利:{5:,.0f}，交易成本:{6:,.0f}'.format(
            self.strategy_name,self.option_portfolio.option_info,self.analysis_dict['total_pnl'],
            self.analysis_dict['delta_pnl'],self.analysis_dict['gamma_pnl'],self.analysis_dict['other_pnl'],
            self.analysis_dict['trading_cost'],self.backtest_df.option_value.iloc[0]))
        plt.show()
        # delta敞口收益分析
        fig3,ax1 = self.init_canvas([0.09,0.05,0.88,0.81])
        ax1.bar(2,self.analysis_dict['delta_pnl_part1'],color='red')
        ax1.bar(4,self.analysis_dict['delta_pnl_part2'],color='orange')
        ax1.bar(6,self.analysis_dict['delta_pnl_part3'],color='green')
        ax1.bar(8,self.analysis_dict['delta_pnl_part4'],color='blue')
        ax1.bar(10,self.analysis_dict['gamma_pnl'],color='purple')
        ax1.bar(12,self.analysis_dict['other_pnl'],color='brown')
        ax1.bar(14,-self.analysis_dict['trading_cost'],color='grey')
        ax1.set_xticks([2,4,6,8,10,12,14])
        ax1.set_xticklabels(['cash_delta>0&△S>0','cash_delta<0&△S>0','cash_delta<0&△S<0','cash_delta>0&△S<0',
                             'gamma_pnl','other_pnl','trading_cost'])
        ax1.set_ylabel('金额/元')
        ax1.set_title('收益拆解，对冲策略:{0:s}\n{1:s}\n 四个累计收益:{2:,.0f} {3:,.0f} {4:,.0f} {5:,.0f}\ngamma收益:{6:,.0f}，其他盈利:{7:,.0f}，交易成本:{8:,.0f}'.format(
            self.strategy_name,self.option_portfolio.option_info,self.analysis_dict['delta_pnl_part1'],
            self.analysis_dict['delta_pnl_part2'],self.analysis_dict['delta_pnl_part3'],self.analysis_dict['delta_pnl_part4'],
            self.analysis_dict['gamma_pnl'],self.analysis_dict['other_pnl'],self.analysis_dict['trading_cost']))
        plt.show()
        # delta敞口收益散点图
        fig4,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        t_data = self.backtest_df.loc[:,['trade_type','delta_pnl_contribution','gamma_pnl','other_pnl']].copy()
        t_data['delta_price'] = self.backtest_df.loc[:,'stock_price'].diff()
        t_data['total_cash_delta'] = self.backtest_df.loc[:,'total_cash_delta'].shift(1)
        t_data['delta_pnl_contribution'] = t_data['delta_pnl_contribution'].astype(float)
        x_min = t_data['total_cash_delta'].min()
        x_max = t_data['total_cash_delta'].max()
        y_min = t_data['delta_price'].min()
        y_max = t_data['delta_price'].max()
        ax1.scatter(t_data.loc[t_data.trade_type==1,'total_cash_delta'],
                    t_data.loc[t_data.trade_type==1,'delta_price'],color='red',
                    label='cash_delta>0&△S>0')
        ax1.scatter(t_data.loc[t_data.trade_type==2,'total_cash_delta'],
                    t_data.loc[t_data.trade_type==2,'delta_price'],color='orange',
                    label='cash_delta<0&△S>0')
        ax1.scatter(t_data.loc[t_data.trade_type==3,'total_cash_delta'],
                    t_data.loc[t_data.trade_type==3,'delta_price'],color='green',
                    label='cash_delta<0&△S<0')
        ax1.scatter(t_data.loc[t_data.trade_type==4,'total_cash_delta'],
                    t_data.loc[t_data.trade_type==4,'delta_price'],color='blue',
                    label='cash_delta>0&△S<0')
        ax1.vlines(0,y_min,y_max)
        ax1.hlines(0,x_min,x_max)
        ax1.set_xlabel('cash_delta敞口')
        ax1.set_ylabel('价格变动')
        ax1.legend(loc='best')
        ax1.set_title('delta敞口收益散点图，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # 1类收益占比分析
        fig5,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.loc[t_data.trade_type==1].index,
                 t_data.loc[t_data.trade_type==1,'delta_pnl_contribution'],
                 color='red',label='cash_delta>0&△S>0类收益贡献比例')
        ax1.set_ylabel('delta收益贡献比例')
        ax1.legend(loc='best')
        ax1.set_title('cash_delta>0&△S>0类收益贡献比例分析，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # 2类收益占比分析
        fig6,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.loc[t_data.trade_type==2].index,
                 t_data.loc[t_data.trade_type==2,'delta_pnl_contribution'],
                 color='orange',label='cash_delta<0&△S>0类收益贡献比例')
        ax1.set_ylabel('delta收益贡献比例')
        ax1.legend(loc='best')
        ax1.set_title('cash_delta<0&△S>0类收益贡献比例分析，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # 3类收益占比分析
        fig7,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.loc[t_data.trade_type==3].index,
                 t_data.loc[t_data.trade_type==3,'delta_pnl_contribution'],
                 color='green',label='cash_delta<0&△S<0类收益贡献比例')
        ax1.set_ylabel('delta收益贡献比例')
        ax1.legend(loc='best')
        ax1.set_title('cash_delta<0&△S<0类收益贡献比例分析，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # 4类收益占比分析
        fig8,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.loc[t_data.trade_type==4].index,
                 t_data.loc[t_data.trade_type==4,'delta_pnl_contribution'],
                 color='blue',label='cash_delta>0&△S<0类收益贡献比例')
        ax1.set_ylabel('delta收益贡献比例')
        ax1.legend(loc='best')
        ax1.set_title('cash_delta>0&△S<0类收益贡献比例分析，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # gamma收益分析
        fig9,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.index,t_data.loc[:,'gamma_pnl'],
                 color='purple',label='gamma日收益')
        ax1.set_ylabel('收益/元')
        ax1.legend(loc='best')
        ax1.set_title('gamma日收益，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        # 其他收益
        fig10,ax1 = self.init_canvas([0.09,0.08,0.88,0.84])
        ax1.plot(t_data.index,t_data.loc[:,'other_pnl'],
                 color='brown',label='其他日收益')
        ax1.set_ylabel('收益/元')
        ax1.legend(loc='best')
        ax1.set_title('其他日收益，对冲策略:{0:s}\n{1:s}'.format(
            self.strategy_name,self.option_portfolio.option_info))
        plt.show()
        if report:
            current_time = str(datetime.now()).replace(':', '：')
            self.check_folder(self.REPORT_FOLDER)
            report_name = os.path.join(self.REPORT_FOLDER,'期权类型：{0:s}，对冲策略：{1:s}_{2:s}.pdf'.format(self.option_portfolio.option_name,self.strategy_name,current_time))
            with PdfPages(report_name) as pdf:
                pdf.savefig(fig1)
                pdf.savefig(fig2)
                pdf.savefig(fig3)
                pdf.savefig(fig4)
                pdf.savefig(fig5)
                pdf.savefig(fig6)
                pdf.savefig(fig7)
                pdf.savefig(fig8)
                pdf.savefig(fig9)
                pdf.savefig(fig10)
        
    def init_analysis(self):
        self.analysis_dict = dict().fromkeys(self.analysis_index)

    @staticmethod
    def init_canvas(rect=[0.05, 0.05, 0.9, 0.9]):
        fig = plt.figure(figsize=(10, 5.7), dpi=300)
        ax = fig.add_axes(rect=rect)
        return fig,ax

    def init_backtest(self):
        self.stock_prices = self.option_portfolio.public_df.loc[:, 'stock_price']
        self.trade_dates = self.option_portfolio.trade_dates
        self.backtest_df = pd.DataFrame(index=self.trade_dates,columns=self.backtest_columns)
        self.backtest_df.loc[:, 'option_value'] = self.option_portfolio.greek_df.loc[:, 'option_value']
        self.backtest_df.loc[:, 'stock_price'] = self.stock_prices
        self.backtest_df.loc[:, 'sigma'] = self.option_portfolio.public_df.loc[:,'sigma']

    @staticmethod
    def cal_MDD(series):
        return np.max(series.cummax()-series)
    
    @staticmethod
    def check_folder(temp_folder):
        if not os.path.isdir(temp_folder):
            BacktestFramework.make_folder(temp_folder)
            
    @staticmethod
    def make_folder(temp_folder):
        if not os.path.isdir(os.path.dirname(temp_folder)):
            BacktestFramework.make_folder(os.path.dirname(temp_folder))
        os.makedirs(temp_folder)


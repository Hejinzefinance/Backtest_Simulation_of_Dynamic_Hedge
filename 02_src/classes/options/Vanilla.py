from .OptionBase import OptionBase
import pandas as pd
import numpy as np
from scipy import stats as st
class VanillaCall(OptionBase):
    def __init__(self):
        super().__init__()

    def calculate_greeks(self):
        self.get_stock_prices()
        vol = self.stock_prices.pct_change().rolling(120).std()*np.sqrt(2*252)
        self.greek_df = pd.DataFrame(index=self.trade_datetimes,columns=self.greek_columns)
        self.greek_df.loc[:,'sigma'] = vol.dropna()
        self.greek_df.loc[:, 'left_days'] = np.linspace(self.datetime_length/2-0.5,0.001,self.datetime_length)
        self.greek_df.loc[:, 'left_times'] = self.greek_df.loc[:, 'left_days']/252
        self.greek_df.loc[:, 'sigma_T'] = self.greek_df.loc[:,'sigma']*np.sqrt(self.greek_df.loc[:, 'left_times'])
        self.greek_df.loc[:, 'stock_price'] = self.stock_prices.loc[self.trade_datetimes]
        self.greek_df.loc[:, 'd1'] = (np.log(self.greek_df.loc[:, 'stock_price']/self.K)+self.r*self.greek_df.loc[:, 'left_times'])/self.greek_df.loc[:,'sigma_T']+0.5*self.greek_df.loc[:,'sigma_T']
        self.greek_df.loc[:, 'd2'] = self.greek_df.loc[:, 'd1']-self.greek_df.loc[:,'sigma_T']
        self.greek_df.loc[:, 'nd1'] = st.norm.pdf(self.greek_df.loc[:, 'd1'])
        self.greek_df.loc[:, 'Nd1'] = st.norm.cdf(self.greek_df.loc[:, 'd1'])
        self.greek_df.loc[:, 'Nd2'] = st.norm.cdf(self.greek_df.loc[:, 'd2'])
        self.greek_df.loc[:, 'delta'] = self.greek_df.loc[:, 'Nd1']
        self.greek_df.loc[:, 'gamma'] = self.greek_df.loc[:, 'nd1']/self.greek_df.loc[:, 'stock_price']/self.greek_df.loc[:, 'sigma_T']
        self.greek_df.loc[:, 'vega'] = self.greek_df.loc[:, 'nd1']*self.greek_df.loc[:, 'stock_price']*np.sqrt(self.greek_df.loc[:, 'left_times'])
        self.greek_df.loc[:, 'theta'] = -self.greek_df.loc[:, 'stock_price']*self.greek_df.loc[:, 'nd1']*self.greek_df.loc[:,'sigma']/2/np.sqrt(self.greek_df.loc[:, 'left_times'])-self.r*self.K*np.exp(-self.r*self.greek_df.loc[:, 'left_times'])*self.greek_df.loc[:, 'Nd2']
        self.greek_df.loc[:, 'option_price'] = self.greek_df.loc[:, 'stock_price']*self.greek_df.loc[:, 'Nd1']-self.K*np.exp(-self.r*self.greek_df.loc[:, 'left_times'])*self.greek_df.loc[:, 'Nd2']
        self.greek_df.loc[:, 'cash_delta'] = self.greek_df.loc[:, 'delta']*self.greek_df.loc[:, 'stock_price']*self.notional/self.start_price
        self.greek_df.loc[:, 'cash_gamma'] = self.greek_df.loc[:, 'gamma']*np.power(self.greek_df.loc[:, 'stock_price'],2)*self.notional/self.start_price/100
        self.greek_df.loc[:, 'pos_vega'] = self.greek_df.loc[:, 'vega']*self.notional/self.start_price
        self.greek_df.loc[:, 'cash_theta'] = self.greek_df.loc[:, 'theta']/252*self.notional/self.start_price
        self.greek_df.loc[:, 'option_value'] = self.greek_df.loc[:, 'option_price']*self.notional/self.start_price

class VanillaPut(OptionBase):
    def __init__(self):
        super().__init__()

    def calculate_greeks(self):
        self.get_stock_prices()
        vol = self.stock_prices.pct_change().rolling(120).std()*np.sqrt(2*252)
        self.greek_df = pd.DataFrame(index=self.trade_datetimes,columns=self.greek_columns)
        self.greek_df.loc[:,'sigma'] = vol.dropna()
        self.greek_df.loc[:, 'left_days'] = np.linspace(self.datetime_length/2-0.5,0.001,self.datetime_length)
        self.greek_df.loc[:, 'left_times'] = self.greek_df.loc[:, 'left_days']/252
        self.greek_df.loc[:, 'sigma_T'] = self.greek_df.loc[:,'sigma']*np.sqrt(self.greek_df.loc[:, 'left_times'])
        self.greek_df.loc[:, 'stock_price'] = self.stock_prices.loc[self.trade_datetimes]
        self.greek_df.loc[:, 'd1'] = (np.log(self.greek_df.loc[:, 'stock_price']/self.K)+self.r*self.greek_df.loc[:, 'left_times'])/self.greek_df.loc[:,'sigma_T']+0.5*self.greek_df.loc[:,'sigma_T']
        self.greek_df.loc[:, 'd2'] = self.greek_df.loc[:, 'd1']-self.greek_df.loc[:,'sigma_T']
        self.greek_df.loc[:, 'nd1'] = st.norm.pdf(self.greek_df.loc[:, 'd1'])
        self.greek_df.loc[:, 'Nd1'] = st.norm.cdf(self.greek_df.loc[:, 'd1'])
        self.greek_df.loc[:, 'Nd2'] = st.norm.cdf(self.greek_df.loc[:, 'd2'])
        self.greek_df.loc[:, 'delta'] = self.greek_df.loc[:, 'Nd1']-1
        self.greek_df.loc[:, 'gamma'] = self.greek_df.loc[:, 'nd1']/self.greek_df.loc[:, 'stock_price']/self.greek_df.loc[:, 'sigma_T']
        self.greek_df.loc[:, 'vega'] = self.greek_df.loc[:, 'nd1']*self.greek_df.loc[:, 'stock_price']*np.sqrt(self.greek_df.loc[:, 'left_times'])
        self.greek_df.loc[:, 'theta'] = -self.greek_df.loc[:, 'stock_price']*self.greek_df.loc[:, 'nd1']*self.greek_df.loc[:,'sigma']/2/np.sqrt(self.greek_df.loc[:, 'left_times'])-self.r*self.K*np.exp(-self.r*self.greek_df.loc[:, 'left_times'])*(self.greek_df.loc[:, 'Nd2']-1)
        self.greek_df.loc[:, 'option_price'] = self.greek_df.loc[:, 'stock_price']*(self.greek_df.loc[:, 'Nd1']-1)-self.K*np.exp(-self.r*self.greek_df.loc[:, 'left_times'])*(self.greek_df.loc[:, 'Nd2']-1)
        self.greek_df.loc[:, 'cash_delta'] = self.greek_df.loc[:, 'delta']*self.greek_df.loc[:, 'stock_price']*self.notional/self.start_price
        self.greek_df.loc[:, 'cash_gamma'] = self.greek_df.loc[:, 'gamma']*np.power(self.greek_df.loc[:, 'stock_price'],2)*self.notional/self.start_price/100
        self.greek_df.loc[:, 'pos_vega'] = self.greek_df.loc[:, 'vega']*self.notional/self.start_price
        self.greek_df.loc[:, 'cash_theta'] = self.greek_df.loc[:, 'theta']/252*self.notional/self.start_price
        self.greek_df.loc[:, 'option_value'] = self.greek_df.loc[:, 'option_price']*self.notional/self.start_price
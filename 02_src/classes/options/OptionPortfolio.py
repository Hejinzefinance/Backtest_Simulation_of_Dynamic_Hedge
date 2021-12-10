from .Vanilla import VanillaCall,VanillaPut
from ..basicData.basicData import BasicData
import pandas as pd
class OptionPortfolio:
    all_trade_dates = BasicData.ALL_TRADE_DATES
    price_dict = BasicData.PRICE_DICT
    def __init__(self):
        self.public_columns = ['sigma','left_days','left_times','sigma_T','stock_price']
        self.greek_columns = ['cash_delta', 'cash_gamma','pos_vega','cash_theta', 'option_value']
        self.option_basket = []
        self.reset()

    def reset(self):
        self.notional = 0
        self.stock_code = None
        self.start_date = None
        self.end_date = None
        self.option_fee = 0
        self.trade_dates = None
        self.trade_datetimes = None

    def add_option(self,option_class,option_position,**option_paras):
        option_dict = dict().fromkeys(['option_obj','option_pos'])
        option_dict['option_obj'] = eval(option_class)()
        option_dict['option_obj'].set_paras_by_dict(option_paras)
        option_dict['option_pos'] = option_position
        option_dict['option_obj'].calculate_greeks()
        if len(self.option_basket)==0:
            self.get_paras_from_current_option(option_dict['option_obj'])
        self.update_paras(option_dict['option_obj'])
        self.option_basket.append(option_dict)

    def add_option_by_dict(self,option_class,option_position,option_paras):
        option_dict = dict().fromkeys(['option_obj','option_pos'])
        option_dict['option_obj'] = eval(option_class)()
        option_dict['option_obj'].set_paras_by_dict(option_paras)
        option_dict['option_pos'] = option_position
        option_dict['option_obj'].calculate_greeks()
        if len(self.option_basket)==0:
            self.get_paras_from_current_option(option_dict['option_obj'])
        self.update_paras(option_dict)
        self.option_basket.append(option_dict)
        self.get_option_name()
    
    def get_option_name(self):
        if len(self.option_basket)==1:
            self.option_name = str(type(self.option_basket[0]['option_obj'])).strip("'>").split('.')[-1]
            self.strike_price = self.option_basket[0]['option_obj'].K

    def get_paras_from_current_option(self,option):
        self.stock_code = option.stock_code
        self.start_date = option.start_date
        self.end_date = option.end_date
        self.trade_dates = option.trade_dates
        self.trade_datetimes = option.trade_datetimes
        self.public_df = pd.DataFrame(index=self.trade_datetimes, columns=self.public_columns)
        self.public_df.loc[:,:] = option.greek_df.loc[:,['sigma','left_days','left_times','sigma_T','stock_price']]
        self.greek_df = pd.DataFrame(0,index=self.trade_datetimes, columns=self.greek_columns)

    def update_paras(self,option):
        self.notional += option['option_obj'].notional
        self.option_fee += option['option_obj'].option_fee
        self.greek_df.loc[:,:] += (option['option_pos'])*option['option_obj'].greek_df.loc[:,['cash_delta', 'cash_gamma','pos_vega','cash_theta', 'option_value']]

    def get_greeks(self):
        return self.greek_df
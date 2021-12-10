import pandas as pd
import pickle
excel_file = '../data/Wind_data/Wind数据.xlsx'
all_trade_dates = list(pd.read_excel(excel_file,sheet_name='交易日',squeeze=True))
with open('../data/all_trade_dates.pickle', 'wb+') as f:
    pickle.dump(all_trade_dates,f)
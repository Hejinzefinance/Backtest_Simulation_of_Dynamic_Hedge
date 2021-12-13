import pandas as pd
import pickle
excel_file = '../data/Wind_data/Wind数据.xlsx'
open_price = pd.read_excel(excel_file,sheet_name='open').set_index('date')
close_price = pd.read_excel(excel_file,sheet_name='close').set_index('date')
price_dict = {'open':open_price,'close':close_price}
with open('../data/price_dict.pickle','wb+') as f:
    pickle.dump(price_dict,f)




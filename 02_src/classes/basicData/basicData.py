import pickle
class BasicData:
    with open('data/all_trade_dates.pickle', 'rb') as f:
        ALL_TRADE_DATES = pickle.load(f)
    with open('data/price_dict.pickle', 'rb') as f:
        PRICE_DICT = pickle.load(f)
    MULTIPLIER = 100
    def __new__(cls):
        return cls

import glob

import pandas as pd

files = glob.glob("*/*.csv")

df_list = [pd.read_csv(file) for file in files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.dropna()
merged_df.to_csv("1merged_file.csv", index=False)

symbols = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
    'SUI/USDT',
    'ADA/USDT',
    'XRP/USDT',
    'DOT/USDT',
    'LINK/USDT',
    'LTC/USDT',
    'AAVA/USDT',
    'UNI/USDT',
    'ACAX/USDT',
    'BCH/USDT',
    'XLM/USDT',
]

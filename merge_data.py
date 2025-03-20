import glob

import pandas as pd

files = glob.glob("*/*.csv")

df_list = [pd.read_csv(file) for file in files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_csv("merged_file.csv", index=False)

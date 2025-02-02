import pandas as pd

data_path = "./data/ufc_fight_stats.csv"
ufc_fight_stats = pd.read_csv(data_path)
print(ufc_fight_stats.head())

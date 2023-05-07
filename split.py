import pandas as pd
import os

df = pd.read_csv("C:/Users/Ethan/exhibhition/train.csv/train.csv")

for _, row in df.iterrows():
  f = row['image_n']
  l = row['target']
  os.replace(f'C:/Users/Ethan/exhibhition/jpeg/train/{f}', f'C:/Users/Ethan/exhibhition/jpeg/train{l}/{f}')
  
import pandas as pd
import numpy as np


a  = pd.read_table('110170109.txt', sep='\s+')
columns = ['Time', '(s)', 'Voltage', '(V)']

df = pd.DataFrame(a.to_numpy().reshape(-1,len(columns)), columns=columns)

# print(df.sample())

time = df['(s)']
voltage = df['Voltage']
Table = np.concatenate((time.T, voltage.T)).reshape((2, -1))
Table[1]= np.round_(Table[1], 6)
pd.DataFrame(Table.T).to_csv("fixed_data.csv")
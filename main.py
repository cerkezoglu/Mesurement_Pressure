import pandas as pd
import numpy as np


a  = pd.read_table('dataset3.txt', sep='\s+')
columns = ['Time', '(s)', 'Voltage', '(V)']

df = pd.DataFrame(a.to_numpy().reshape(-1,len(columns)), columns=columns)

# print(df.sample())

time = df['(s)']
voltage = df['Voltage']
Table = np.concatenate((time.T, voltage.T)).reshape((2, -1))
Table[1] = np.round_(Table[1], 6)
Table[0] = np.round_(Table[0], 3)
pd.DataFrame(Table.T).to_csv("dataset3.csv", header= ["Time", "Voltage"])
import pandas as pd


a  = pd.read_table('110170109.txt', sep='\s+')
columns = ['Time', '(s)', 'Voltage', '(V)']

df = pd.DataFrame(a.to_numpy().reshape(-1,len(columns)), columns=columns)

print(df.sample())

print(df['Voltage'][2])
# for x in df['Voltage']:
#     print(x)
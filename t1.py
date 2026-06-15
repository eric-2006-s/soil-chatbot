import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_excel("Koppal_Union_Plot_Soil.xlsx")

print(df)
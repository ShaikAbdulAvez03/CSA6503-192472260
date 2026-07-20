import pandas as pd

data = {"Name": ["Ram", "Sam"], "Marks": [85, 90]}
df = pd.DataFrame(data)
print(df["Name"])
print(df["Marks"])
print(df.describe())
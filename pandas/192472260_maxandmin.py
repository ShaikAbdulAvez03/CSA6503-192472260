import pandas as pd

data = {"Marks": [60, 75, 90, 85]}
df = pd.DataFrame(data)
print("Maximum:", df["Marks"].max())
print("Minimum:", df["Marks"].min())
print(df)
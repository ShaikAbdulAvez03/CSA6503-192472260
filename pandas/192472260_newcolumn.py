import pandas as pd

data = {"Name": ["A", "B"], "Marks": [70, 80]}
df = pd.DataFrame(data)
df["Grade"] = ["B", "A"]
print(df)
print(df.columns)
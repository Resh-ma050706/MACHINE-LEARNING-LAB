import pandas as pd
import matplotlib.pyplot as plt
file_path = r"C:\Users\ASUS\Downloads\archive (2)\data.csv"
df = pd.read_csv(file_path, encoding="latin1")
print("S.RESHMA 24BAD097")
print("Dataset loaded successfully!")
print(df.head())
print(df.info())
df['Sales'] = df['Quantity'] * df['UnitPrice']
top_products = (
    df.groupby('Description')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
plt.figure()
top_products.plot(kind='bar')
plt.title("Top 10 Products by Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

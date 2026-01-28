import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = r"C:\Users\ASUS\Downloads\exp1.3\Housing.csv"
df = pd.read_csv(file_path)
print("S.RESHMA 24BAD097")
print("Dataset loaded successfully!\n")
#  Inspect columns
print("----- COLUMNS -----")
print(df.columns)
print("\n----- HEAD -----")
print(df.head())
print("\n----- INFO -----")
df.info()
print("\n----- DESCRIBE -----")
print(df.describe())
#  Detect and summarize missing values
print("\n----- MISSING VALUES -----")
missing_values = df.isnull().sum()
print(missing_values)
print("\n----- TOTAL MISSING VALUES -----")
print("Total missing values:", df.isnull().sum().sum())
# Display missing values percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\n----- MISSING VALUES PERCENTAGE (%) -----")
print(missing_percent)
#  Select only numerical columns
num_df = df.select_dtypes(include=[np.number])
print("\n----- NUMERICAL COLUMNS -----")
print(num_df.columns)
target_col = "price"  # change to "Price" if needed
if target_col in df.columns:
    for col in num_df.columns:
        if col != target_col:
            plt.figure()
            plt.scatter(df[col], df[target_col])
            plt.title(f"{target_col} vs {col}")
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.tight_layout()
            plt.show()
else:
    print(f"\n Target column '{target_col}' not found.")
    print("Available columns are:", df.columns)
#  Heatmap for correlation between numerical features
corr_matrix = num_df.corr()
plt.figure(figsize=(10, 6))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()
#  Interpretation trends for price prediction
print("\n----- TOP CORRELATED FEATURES WITH PRICE -----")
if target_col in num_df.columns:
    corr_with_price = corr_matrix[target_col].sort_values(ascending=False)
    print(corr_with_price)
else:
    print("Price column not found in numerical dataset.")

import pandas as pd
import matplotlib.pyplot as plt
file_path = r"C:\Users\ASUS\Downloads\exp1.2\diabetes.csv"
df = pd.read_csv(file_path)
print("Dataset loaded successfully!\n")
print("----- HEAD -----")
print(df.head())
print("\n----- INFO -----")
df.info()
print("\n----- DESCRIBE -----")
print(df.describe())
print("\n----- MISSING VALUES CHECK (Zeros as missing) -----")
columns_with_zero_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zero_missing:
    print(col, ":", (df[col] == 0).sum())
df[columns_with_zero_missing] = df[columns_with_zero_missing].replace(0, pd.NA)
print("\nMissing values after replacement:")
print(df.isna().sum())
plt.figure()
plt.hist(df['Glucose'].dropna(), bins=20)
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.show()
plt.figure()
plt.boxplot(df['Age'].dropna())
plt.title("Age Distribution of Patients")
plt.ylabel("Age")
plt.show()
print("\n----- PATTERN ANALYSIS -----")
print("Average values grouped by Diabetes Outcome:\n")
print(df.groupby('Outcome').mean())
print("\nOutcome Legend:")
print("0 → Non-Diabetic")
print("1 → Diabetic")
print("\nAnalysis completed successfully!")



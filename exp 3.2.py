import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
print("S.RESHMA 24BAD097")
df = pd.read_csv(r"C:\Users\ASUS\Downloads\exp3.2\auto-mpg.csv")
print("Dataset Shape:", df.shape)
print(df.head())
df.replace('?', np.nan, inplace=True)
# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])
X = df[['horsepower']]
y = df['mpg']
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
degrees = [2, 3, 4]
results = {}
train_errors = []
test_errors = []
plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Actual Data", alpha=0.5)
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    results[deg] = [mse, rmse, r2]
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
    X_range_scaled = scaler.transform(X_range)
    X_range_poly = poly.transform(X_range_scaled)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, label=f"Degree {deg}")
results_df = pd.DataFrame(results, index=["MSE", "RMSE", "R2"]).T
print("\nPerformance Comparison:\n", results_df)
poly4 = PolynomialFeatures(degree=4)
X_poly_train4 = poly4.fit_transform(X_train)
X_poly_test4 = poly4.transform(X_test)
ridge = Ridge(alpha=1.0)
ridge.fit(X_poly_train4, y_train)
y_pred_ridge = ridge.predict(X_poly_test4)
print("\nRidge Regression Results:")
print("MSE :", mean_squared_error(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R2  :", r2_score(y_test, y_pred_ridge))
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Curve Fitting")
plt.legend()
plt.show()
plt.figure(figsize=(8,5))
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Training vs Testing Error Comparison")
plt.legend()
plt.show()
plt.figure(figsize=(8,5))
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Overfitting and Underfitting Demonstration")
plt.legend()
plt.show()

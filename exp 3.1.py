import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
print("S.RESHMA 24BAD097")
df = pd.read_csv(r"C:\Users\ASUS\Downloads\exp3.1\StudentsPerformance.csv")
print("Dataset Shape:", df.shape)
print(df.head())
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['race/ethnicity'] = encoder.fit_transform(df['race/ethnicity'])
df['parental level of education'] = encoder.fit_transform(df['parental level of education'])
df['lunch'] = encoder.fit_transform(df['lunch'])
df['test preparation course'] = encoder.fit_transform(df['test preparation course'])
df['final_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df.fillna(df.mean(), inplace=True)
X = df[['gender',
        'parental level of education',
        'test preparation course',
        'lunch']]
y = df['final_score']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Influence:\n", coeff_df)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
print("\nRidge R²:", ridge.score(X_test, y_test))
print("Lasso R²:", lasso.score(X_test, y_test))
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted")
plt.show()
coeff_df.plot(kind='bar')
plt.title("Feature Importance")
plt.show()
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

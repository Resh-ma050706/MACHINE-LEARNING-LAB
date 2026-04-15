print("S.RESHMA 24BAD097")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
file_path = r"C:\Users\ASUS\Downloads\exp 8.2\Iris.csv"
df = pd.read_csv(file_path)
print("First 5 rows:")
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())
df['Species'] = df['Species'].astype('category').cat.codes
df = df.drop('Id', axis=1)
X = df.drop('Species', axis=1)
print("\nFeatures Used:")
print(X.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")
pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X_scaled)
pca_3 = PCA(n_components=3)
X_3d = pca_3.fit_transform(X_scaled)
plt.figure()
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.title("Scree Plot")
plt.xlabel("Components")
plt.ylabel("Variance")
plt.grid()
plt.show()
cum_var = np.cumsum(explained_variance)
plt.figure()
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.title("Cumulative Variance")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['Species'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA")
plt.grid()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=df['Species'])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA")
plt.show()
print("\nCumulative Variance:")
for i, val in enumerate(cum_var):
    print(f"{i+1} components: {val:.4f}")
optimal = np.argmax(cum_var >= 0.95) + 1
print(f"\nOptimal components (>=95% variance): {optimal}")
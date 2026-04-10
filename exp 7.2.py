print("S.RESHMA 24BAD097")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv(r"C:\Users\ASUS\Downloads\exp 7\Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)
probs = gmm.predict_proba(X_scaled)
gmm_labels = np.argmax(probs, axis=1)
df['GMM_Cluster'] = gmm_labels
plt.figure()
for i in range(5):
    plt.scatter(
        df[df['GMM_Cluster'] == i]['Annual Income (k$)'],
        df[df['GMM_Cluster'] == i]['Spending Score (1-100)'],
        label=f'Cluster {i}'
    )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('GMM Clusters')
plt.legend()
plt.grid(True)
plt.show()
x = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)
X_grid, Y_grid = np.meshgrid(x, y)
grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
Z = -gmm.score_samples(grid)
Z = Z.reshape(X_grid.shape)
plt.figure()
plt.contour(X_grid, Y_grid, Z)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
plt.xlabel('Scaled Income')
plt.ylabel('Scaled Spending')
plt.title('GMM Contour')
plt.show()
plt.figure()
plt.hist(probs.max(axis=1), bins=10)
plt.xlabel('Max Probability')
plt.ylabel('Frequency')
plt.title('Cluster Probability')
plt.show()
print("Log-Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))
print("Silhouette Score:", silhouette_score(X_scaled, gmm_labels))
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
plt.title('K-Means')
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
plt.title('GMM')
plt.show()
print(df.groupby('GMM_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
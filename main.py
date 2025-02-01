from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(42)
training_hours = np.random.randint(10, 40, 100)
game_points = 3*training_hours + np.random.randint(10, 30, 100)
data = {
    'Training Hours': training_hours,
    'Game Points': game_points
}
df = pd.DataFrame(data)
X = df[['Training Hours']]
Y = df[['Game Points']]
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, Y_pred, color="red")
plt.title("Linear Regression")
plt.xlabel("Training Hours")
plt.ylabel("Game Points")
plt.show()

centers = [(8, 8), (5, 5), (2, 2)]  # Центри для трьох груп
cluster_std = [1.5, 1.0, 1.2]  # Різна дисперсія для реалістичності

data, _ = make_blobs(n_samples=100, centers=centers, cluster_std=cluster_std, random_state=42)
speed = data[:, 0]
stamina = data[:, 1]

data = {
    'Speed': speed,
    'Stamina': stamina
}
df_players = pd.DataFrame(data)

# Виконуємо кластеризацію K-Means на 3 групи
kmeans = KMeans(n_clusters=3, random_state=42)
df_players['Cluster'] = kmeans.fit_predict(df_players[['Speed', 'Stamina']])
centroids = kmeans.cluster_centers_

# Візуалізація результатів
plt.scatter(df_players['Speed'], df_players['Stamina'], c=df_players['Cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='x', label='Centroids')
plt.xlabel('Speed')
plt.ylabel('Stamina')
plt.title('Player Clustering')
plt.legend()
plt.show()

# Вивід розподілу гравців за категоріями
print(df_players.head())

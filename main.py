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
print(df)
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

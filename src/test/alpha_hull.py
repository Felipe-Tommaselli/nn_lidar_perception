import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

# Criando dados de exemplo
x = np.random.rand(100)
y = np.random.rand(100)

# Ordenando x e y em ordem crescente
x, y = zip(*sorted(zip(x, y)))

# Encontrando os vizinhos mais próximos
neighbors = NearestNeighbors(n_neighbors=10).fit(np.column_stack((x, y)))
distances, indices = neighbors.kneighbors(np.column_stack((x, y)))

# Criando triângulos de Delaunay
tri = Delaunay(np.column_stack((x, y)))

# Definindo o valor do alpha
alpha = 0.05

# Encontrando o conjunto de pontos que pertencem ao Alpha Hull
alpha_hull_points = set()
for simplex in tri.simplices:
    if np.max(distances[simplex]) < alpha:
        alpha_hull_points.update(simplex)

# Plotando os pontos do Alpha Hull
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y, alpha=0.5)
ax.scatter(np.array(x)[list(alpha_hull_points)], np.array(y)[list(alpha_hull_points)], c='r', alpha=0.8, s=20)
plt.show()

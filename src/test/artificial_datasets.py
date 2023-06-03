import numpy as np
import matplotlib.pyplot as plt

# Configurações
image_size = 224  # Tamanho da imagem
divider = 30  # Espaçamento entre as retas
num_points = 500  # Número total de pontos
max_points_between = 20  # Número máximo de pontos entre as retas

# Equação da reta para as retas paralelas
x1 = image_size // 2 - divider // 2  # Coordenada x para a primeira reta
x2 = image_size // 2 + divider // 2  # Coordenada x para a segunda reta

# Gerar coordenadas aleatórias para os pontos
x_coords = []
y_coords = []

# Gerar pontos entre as retas
for i in range(num_points):
    if i == 0 or (x1 + max_points_between < x_coords[i-1] < x2 - max_points_between):
        # Gerar um número reduzido de pontos entre as retas
        x = np.random.randint(x1, x2)
    else:
        # Gerar pontos aleatoriamente fora das retas
        if np.random.rand() < 0.5:
            x = np.random.randint(0, x1)
        else:
            x = np.random.randint(x2+1, image_size)
    
    y = np.random.randint(0, image_size)
    x_coords.append(x)
    y_coords.append(y)

# Criar a figura e o eixo
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, image_size)
ax.set_ylim(0, image_size)

# Desenhar as retas
ax.plot([x1, x1], [0, image_size], 'r')
ax.plot([x2, x2], [0, image_size], 'r')

# Desenhar os pontos
ax.scatter(x_coords, y_coords, s=14, c='black')

# Remover as bordas e ticks dos eixos
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')


# Salvar a imagem
plt.savefig('dataset_image.png', bbox_inches='tight', pad_inches=0, dpi=300)

# Mostrar a imagem na tela
plt.show()
plt.close()

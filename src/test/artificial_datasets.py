import numpy as np
import matplotlib.pyplot as plt
import math

# Configurações
image_size = 224  # Tamanho da imagem
divider = 80  # Espaçamento entre as retas
pivot = (112, 224)

# Equação da reta para as retas paralelas
x1 = image_size // 2 - divider // 2  # Coordenada x para a primeira reta
x2 = image_size // 2 + divider // 2  # Coordenada x para a segunda reta


# Gerar coordenadas aleatórias para os pontos
x_coords = []
y_coords = []

# Gerar pontos entre as retas
for _ in range(30):
    x = np.random.choice([np.random.randint(x1, x1+15), np.random.randint(x2-15, x2)])
    y = np.random.randint(0, image_size//2)
    x_coords.append(x)
    y_coords.append(y)

# Gerar pontos nos 20 primeiros pixels de fora de cada reta
for _ in range(90):
    x = np.random.choice([np.random.randint(x1-15, x1), np.random.randint(x2, x2+15)])
    y = np.random.randint(0, image_size//2)
    x_coords.append(x)
    y_coords.append(y)

# Gerar pontos nos 20 primeiros pixels de fora de cada reta
for _ in range(25):
    x = np.random.choice([np.random.randint(x1-15, x1), np.random.randint(x2, x2+15)])
    y = np.random.randint(image_size//2, image_size)
    x_coords.append(x)
    y_coords.append(y)

# Gerar pontos entre os 20 primeiros pixels e os 40 primeiros pixels
for _ in range(30):
    x = np.random.choice([np.random.randint(x1-40, x1-15), np.random.randint(x2+15, x2+40)])
    y = np.random.randint(0, image_size//3)
    x_coords.append(x)
    y_coords.append(y)

# Gerar pontos entre os pixels nos pontos mais distantes das retas
for _ in range(20):
    x = np.random.choice([np.random.randint(0, 30), np.random.randint(image_size - 30, image_size)])
    y = np.random.randint(0, image_size//2)
    x_coords.append(x)
    y_coords.append(y)
    

# Gerar pontos aleatórios ao longo da imagem
for _ in range(5):
    x = np.random.randint(0, image_size)
    y = np.random.randint(0, image_size)
    x_coords.append(x)
    y_coords.append(y)


# Definir coordenadas iniciais das retas
start_x1 = pivot[0] - 40
start_y1 = 0
start_x2 = pivot[0] + 40
start_y2 = 0

# Função para rotacionar as retas
def rotate_lines(angle):
    # Converter ângulo para radianos
    theta = np.deg2rad(angle)
    
    # Calcular seno e cosseno do ângulo
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Rotacionar as coordenadas dos pontos
    end_x1 = pivot[0] + cos_theta * (start_x1 - pivot[0]) - sin_theta * (start_y1 - pivot[1])
    end_y1 = pivot[1] + sin_theta * (start_x1 - pivot[0]) + cos_theta * (start_y1 - pivot[1])
    
    end_x2 = pivot[0] + cos_theta * (start_x2 - pivot[0]) - sin_theta * (start_y2 - pivot[1])
    end_y2 = pivot[1] + sin_theta * (start_x2 - pivot[0]) + cos_theta * (start_y2 - pivot[1])
    
    # Criar a figura e o eixo
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_title(f'Angle: {angle}°')
    
    # Desenhar as retas rotacionadas
    ax.plot([start_x1, end_x1], [start_y1, end_y1], 'r')
    ax.plot([start_x2, end_x2], [start_y2, end_y2], 'g')
    
    # Desenhar os pontos
    ax.scatter(x_coords, y_coords, s=80, c='black')
    
    # Remover as bordas e ticks dos eixos
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Salvar a imagem
    plt.savefig(f'dataset_image.png', bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Mostrar a imagem na tela
    plt.show()
    
    plt.close()


# Iterar pelos ângulos desejados
for angle in range(0, 360, 10):
    rotate_lines(angle)
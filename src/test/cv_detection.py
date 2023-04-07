import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# roda os elements na pasta points em uma lista
points = os.listdir('points')

for point in points:
    # Carregar a imagem em escala de cinza
    img = cv2.imread(f'points/{point}', cv2.IMREAD_GRAYSCALE)

    # Criar a figura com subplots
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    fig.suptitle(f'Imagem {point}')

    # Plotar a imagem original
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Imagem original')

    #* Aplicar um filtro de suavização na imagem para remover ruído
        #* Gaussian Blur: Espalha mais e perde a informaçõa do ponto 
        #* Bilateral Filter: Espalha menos e mantém a informação do ponto
        #* Blur: Espalha menos e pondera a informação (efeito de chacoalho)

    img_blur = cv2.GaussianBlur(img, (121, 121), sigmaX=0, sigmaY=80, borderType=cv2.BORDER_DEFAULT)
    # img_blur = cv2.bilateralFilter(img, d=45, sigmaColor=275, sigmaSpace=275)
    # img_blur = cv2.blur(img, (45, 45))
    axs[0, 1].imshow(img_blur, cmap='gray')
    axs[0, 1].set_title('Imagem suavizada')

    # Dilatar a imagem para preencher os buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 10)) # kernel retangular 10x30
    img_dilated = cv2.dilate(img_blur, kernel, iterations=3)
    axs[0, 2].imshow(img_dilated, cmap='gray')
    axs[0, 2].set_title('Imagem Dilatada')

    # Realizar uma binarização na imagem
    ret, thresh = cv2.threshold(img_dilated, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    axs[0, 3].imshow(thresh, cmap='gray')
    axs[0, 3].set_title('Imagem binarizada')

    # Encontrar todos os contornos presentes na imagem
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = np.zeros_like(img)
    cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 2)
    axs[1, 0].imshow(contours_img, cmap='gray')
    axs[1, 0].set_title('Contornos')

    # Selecionar o contorno de maior área
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_img = np.zeros_like(img)
    cv2.drawContours(largest_contour_img, [largest_contour], -1, (255, 255, 255), 2)
    axs[1, 1].imshow(largest_contour_img, cmap='gray')
    axs[1, 1].set_title('Contorno de maior área')

    # Desenhar o contorno na imagem com blur
    img_contours = cv2.drawContours(img_blur, [largest_contour], -1, (0, 255, 0), 3)
    axs[1, 2].imshow(img_contours, cmap='gray')
    axs[1, 2].set_title('Contorno desenhado')

    # Desenhar o contorno selecionado na imagem original
    img_contours = cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 3)
    axs[1, 3].imshow(img_contours, cmap='gray')
    axs[1, 3].set_title('Contorno desenhado')

    # Mostrar a figura com subplots
    plt.show()

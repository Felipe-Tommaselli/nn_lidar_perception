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
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Imagem {point}')

    # inverter a imagem 
    img = cv2.bitwise_not(img)

    # Plotar a imagem original
    axs[0, 0].imshow(cv2.bitwise_not(img), cmap='gray')
    axs[0, 0].set_title('Imagem original')

    #* Aplicar um filtro de suavização na imagem para remover ruído
        #* Gaussian Blur: Espalha mais e perde a informaçõa do ponto 
        #* Bilateral Filter: Espalha menos e mantém a informação do ponto
        #* Blur: Espalha menos e pondera a informação (efeito de chacoalho)

    # Dilatar a imagem para preencher os buracos
    img_blur = cv2.blur(img, (15, 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)) # kernel retangular
    img_dilated = cv2.dilate(img_blur, kernel, iterations=1)
    axs[0, 1].imshow(cv2.bitwise_not(img_dilated), cmap='gray')
    axs[0, 1].set_title('Imagem Dilatada')

    img_blur = cv2.GaussianBlur(img_dilated, (87, 87), sigmaX=0, sigmaY=60, borderType=cv2.BORDER_DEFAULT)
    # img_blur = cv2.bilateralFilter(img, d=45, sigmaColor=275, sigmaSpace=275)
    # img_blur = cv2.blur(img, (45, 45))
    axs[0, 2].imshow(cv2.bitwise_not(img_blur), cmap='gray')
    axs[0, 2].set_title('Imagem suavizada')

    # Realizar uma erosão para remover pontos isolados e pequenas regiões
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 40))
    img_eroded = cv2.erode(img_blur, kernel_erode, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)) # kernel retangular
    img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
    axs[0, 3].imshow(cv2.bitwise_not(img_dilated), cmap='gray')
    axs[0, 3].set_title('Imagem Erodida')

    # Realizar uma binarização na imagem
    ret, thresh = cv2.threshold(img_eroded, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    axs[0, 4].imshow(cv2.bitwise_not(thresh), cmap='gray')
    axs[0, 4].set_title('Imagem binarizada')

    thresh = cv2.bitwise_not(thresh)
    axs[1, 0].imshow(cv2.bitwise_not(thresh), cmap='gray')
    axs[1, 0].set_title('Imagem binarizada')


    # Encontrar todos os contornos presentes na imagem
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = np.zeros_like(img)
    cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 2)
    axs[1, 1].imshow(contours_img, cmap='gray')
    axs[1, 1].set_title('Contornos')

    # Selecionar o contorno de maior área
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_img = np.zeros_like(img)
    cv2.drawContours(largest_contour_img, [largest_contour], -1, (255, 255, 255), 2)
    axs[1, 2].imshow(largest_contour_img, cmap='gray')
    axs[1, 2].set_title('Contorno de maior área')

    # Desenhar o contorno na imagem com blur
    img_contours = cv2.drawContours(cv2.bitwise_not(img_blur), [largest_contour], -1, (0, 255, 0), 3)
    axs[1, 3].imshow(img_contours, cmap='gray')
    axs[1, 3].set_title('Contorno desenhado')

    # Desenhar o contorno selecionado na imagem original
    img_contours = cv2.drawContours(cv2.bitwise_not(img), [largest_contour], -1, (0, 255, 0), 3)
    axs[1, 4].imshow(img_contours, cmap='gray')
    axs[1, 4].set_title('Contorno desenhado')

    # Mostrar a figura com subplots
    plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Exibir imagens em grid
from matplotlib import pyplot as plt

# Mostrar imagens em forma de grade
def showImages(imgsArray, titleArray, size, grid=(1,1)):
    """ 
        imgsArray: Um array contendo as imagens que serão exibidas.
        titlesArray: título para a imagem correspondente em imgsArray.
        size: define o tamanho da figura geral que conterá as imagens.
        grid=(1,1): Uma tupla que define a organização da grade (linhas, colunas). 
            O (1,1), o que significa que exibirá apenas uma imagem."""
    
    y, x = grid

    # Cria uma figura com  linhas e colunas
    #fig: A figura inteira.
    # axes: Um array (ou um único objeto de eixo) que contém grade onde as imagens serão desenhadas.
    fig, axes = plt.subplots(y, x, figsize=size)
    # .ravel() array achatado
    # if criar mais de um subplot, axes será um array NumPy multidimensional. 
    # else (se criar apenas um subplot (o padrão), axes não será um array), np.array() garante que ele seja colocado dentro de um array NumPy
    axes = axes.ravel() if isinstance(axes, np.ndaray) else np.array([axes])

    if len(imgsArray) != len(titleArray):
        print("ERROR: O número de imagens e títulos deve ser o mesmo")
        return
    
    # Loop de exibição
    for idx, (img, title) in enumerate(zip(imgsArray, titleArray)):   #zip(imgsArray, titlesArray): Permite que o loop for acesse uma imagem e seu título correspondente a cada iteração.
        if len(img.shape)== 2: # A imagem é tons de cinza
            axes[idx].imshow(img, cmap = 'gray')
        else: # A imagem é RGB
            axes[idx].imshow(img)
        axes[idx].set_title(title, fontdict={'fontsize': 18, 'fontweight':'medium'}, pad=10) # Configuração do Título
        if len(title) == 0:
            axes[idx].axis('off')

    plt.tight_layout()  # Ajusta automaticamente o Layout para evitar sobreposição
    plt.show




# Para exibir uma única imagem:
#showImages([img1], ["Título 1"], size(10,10), gird(1,1))

# Para exibir duas imagens lado a lado:
#showImages([img1, img2], ["Título 1", Título 2"], size(10,10), gird(1,2))

# Para exibir quatro imagens lem uma grade 2x2:
#showImages([img1, img2, img3, img4], ["Título 1", Título 2", "Título 3", Título 4"], size(10,10), gird(2,2))





# ENCONTRAR CONTORNOS DE UMA IMAGEM BINÁRIA ---------------------------------------------------------------------------------------------------------------------------------------
img = cv2.imread('contours_ex1'.png)
assert img is not None, "file could not be read, check with os.path.exists()"  # Verifica se a imagem foi carregada corretamente
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERTE a imagem BGR para escala de cinza
ret, thresh = cv2.threshold(img_gray, 127, 255,0) # Aplica um threshould para binarizar a imagem

"""" 
Gera os contornos e a hierarquia. 
Contours é uma lista em Python de todos os contornos da imagem. Cada contorno individual é um array Numpy de coordenadas (x,y) dos pontos de contorno do objeto.
Hierarquia é uma matriz que descreve a hierarquia dos contornos
    *contornos, hierarquia = cv.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)*
        o primeiro [thresh] é a imagem de origem
        o segundo [cv2.RETR_TREE] é o modo de recuperação de contornos 
        o terceiro [cv.CHAIN_APPROX_SIMPLE] é o método de aproximação de contornos.  
         cv.CHAIN_APPROX_SIMPLE remove todos os pontos redundantes e compacta o contorno, economizando memória
"""

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours, hierarchy)
# Apenas 1 contorno. Não há relações hierárquica( pai, filho, p´roximo, anterior)
# Contorno. Os valores -1 indicam a ausencia de outros contornos.



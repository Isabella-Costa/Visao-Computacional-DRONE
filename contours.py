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
    axes = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

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
img = cv2.imread('imagens/contours_ex1.png')
assert img is not None, "file could not be read, check with os.path.exists()"  # Verifica se a imagem foi carregada corretamente
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERTE a imagem BGR para escala de cinza
ret, thresh = cv2.threshold(img_gray, 127, 255,0) # Aplica um threshould para binarizar a imagem

"""" 
Gera os contornos e a hierarquia. 
Contours é uma lista de todos os contornos da imagem. Cada contorno individual é um array Numpy de coordenadas (x,y) dos pontos de contorno do objeto.
Hierarquia é uma matriz que descreve a hierarquia dos contornos
    *contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)*
        o primeiro [thresh] é a imagem de origem/ linearizada
        o segundo [cv2.RETR_TREE] é o modo de recuperação de contornos 
        o terceiro [cv2.CHAIN_APPROX_SIMPLE] é o método de aproximação de contornos.  
         cv2.CHAIN_APPROX_SIMPLE remove todos os pontos redundantes e compacta o contorno, economizando memória
"""

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #testar se esssa aproximação é a melhor opção/ INTERPOLAÇÃO

print(contours, hierarchy)
# Apenas 1 contorno. Não há relações hierárquica [Próximo, Anterior, Primeiro_Filho, Pai]
# Contorno. Os valores -1 indicam a ausencia de outros contornos.


# DESENHAR CONTORNO --------------------------------------------------------------------------------------------------------------------------------------------------

'''
A função cv2.drawContours é usada para desenhar os contornos encontrados em uma imagem.
    - O primeiro argumento [img] é a imagem onde se quer que os contornos apareçam
    - O segundo argumento [contours] é a lista de todos os contornos que encontrados com cv.findContours 
    - O terceiro argumento, -1, significa que se deseja desenhar todos os contornos na lista. 
        Se quisesse desenhar apenas um contorno específico, se coloca o indíce do retorno.
    - O quarto argumento [(255, 0, 0)] define a o cor do contorno matplot(RGB)
    - O quinto argumento [3] define a expessura do contorno em 3 pixels

    cv2.drawContours	(InputOutputArray	image,
                         InputArrayOfArrays	contours,
                         int	contourIdx,
                         const Scalar &	color,
                         int	thickness = 1,
                         int	lineType = LINE_8,
                         InputArray	hierarchy = noArray(),
                         int	maxLevel = INT_MAX,
                         Point	offset = Point() )
'''

#cv2.drawContours(img, contours, -1, (255, 0, 0), 3) 
        #não necessariamente scale gray, mas draw.contourns é comum ser colorida(nosso caso não é) 
        #modifica a img original, ideal criar um cópia da img
#showImages([img_gray, img], ["Original", "Exemplo Contorno"], size=(7,7), grid=(1,2)) #RGB 


print(f"Total de contornos: {len(contours)}")

#Criar uma lista de imagens com cada contorno desenhado separadamente
contour_images = []
titles = []

for i, contour in enumerate(contours):
    img_contour = np.zeros_like(img) #Cria uma imagem preta do mesmo tamanho da original
    cv2.drawContours(img_contour, contours, i, (255,255,255), 3)
    contour_images.append(img_contour)
    titles.append(f"Contorno {i}")


num_images = len(contours)
grid_size = (2,4)
showImages([img], ["Original"], size=(4,4), grid= (1,1))
showImages(contour_images, titles, size=(15, num_images), grid=grid_size)

print(hierarchy)
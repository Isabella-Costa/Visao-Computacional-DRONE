import cv2

# Carrega a imagem de um arquivo  -> imagem = cv2.imread(caminho_para_imagem, flag) / flag (opcional): define como a imagem deve ser lida
    # cv2.IMREAD_COLOR (ou 1): Carrega a imagem colorida. Qualquer transparência é ignorada 
    # cv2.IMREAD_GRAYSCALE (ou 0): Carrega a imagem em tons de cinza
img = cv2.imread("OpenCV/pic.jpg", cv2.IMREAD_COLOR) 
img_cinza = cv2.imread("OpenCV/img/pic.jpg", cv2.IMREAD_GRAYSCALE) 


# DADOS DE REDIMENCIONAMENTO DA IMAGEM 
escala_per = 10

largura = int(img.shape[1]*escala_per/100)
altura = int(img.shape[0]*escala_per/100)
#-------------------------------------------------------------------
largura_cinza = int(img_cinza.shape[1]*escala_per/100)
altura_cinza = int(img_cinza.shape[0]*escala_per/100)
#------------------------------------------------------------------
# .shape[] retorna uma tupla.
    # image.shape[0]: Retorna a altura .
    # image.shape[1]: Retorna a largura .
    # image.shape[2]: Retorna o número de canais .

dim = (largura, altura)
dim_cinza = (largura_cinza, altura_cinza)


# Altera o tamanho uma imagem  -> cv2.resize(imagem_original, novo_tamanho, interpolation=metodo)
    # cv2.INTER_AREA: Reduzir o tamanho da imagem.
    # cv2.INTER_CUBIC: Ampliar a imagem. (menos "pixelado").
    # cv2.INTER_LINEAR: É o padrão da função. 
img_redimensionada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
img_redimensionada_cinza = cv2.resize(img_cinza, dim, interpolation = cv2.INTER_AREA) 



# Exibe uma imagem em uma janela
cv2.imshow("Imagem", img_redimensionada) 
cv2.imshow("ImagemCINZA", img_redimensionada_cinza) 


cv2.waitKey(0) # Espera por um determinado tempo (em milissegundos) até que qualquer tecla do teclado seja pressionada
cv2.destroyAllWindows() # Fecha todas as janelas que foram criadas pela OpenCV durante a execução do script.



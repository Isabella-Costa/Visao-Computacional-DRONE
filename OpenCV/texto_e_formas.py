import cv2
import numpy as np  # Funções matemáticas de alto nível para operar arrays/matrizes


# CRIA IMAGEM ------------------------------------------------------------------------------------------------------------------------------------------
# novo_array = np.zeros(shape, dtype=float)
    #shape (Obrigatório): Uma tupla de números inteiros que define as dimensões do array
    #dtype (Opcional): O tipo de dado dos elementos no array. / O padrão é float64
    # Para imagens, o tipo de dado correto é quase sempre np.uint8. 
    # |__| Este é um inteiro de 8 bits sem sinal, que pode armazenar valores de 0 a 255, correspondendo exatamente às intensidades de cor em uma imagem.
    #np.zeros((altura, largura, canais),dtype='uint8')
imagem = np.zeros((400,400,3), dtype='uint8')


"""img_cinza = cv2.imread("OpenCV/img/pic.jpg", cv2.IMREAD_GRAYSCALE) 
escala_per = 10
largura = int(img_cinza.shape[1]*escala_per/100)
altura= int(img_cinza.shape[0]*escala_per/100)
dim = (largura, altura)
imagem= cv2.resize(img_cinza, dim, interpolation = cv2.INTER_AREA) """


# Desenha uma linha reta em uma imagem __________________________________________________________________________________________________________________
# cv2.line(imagem, ponto_inicial, ponto_final, cor(B,G,R), espessura)
        #cv2.line(imagem, (0, 170), (400, 170), (0,255,0), 2)


# Desenha um retângulo em uma imagem __________________________________________________________________________________________________________________
# cv2.rectangle(imagem, ponto1, ponto2, cor(B,G,R), espessura)
        #cv2.rectangle(imagem, (30,30), (200, 200), (0,255,0), 3)


# Desenha um circulo em uma imagem __________________________________________________________________________________________________________________
# cv2.circle(imagem, centro, raio, cor, espessura)
        #cv2.circle(imagem, (200,200), 100, (255,0, 0), 3)



# TEXTO  ___________________________________________________________________________________________________________________________________________
#cv2.putText(imagem, texto, origem, fonte, escala, cor, espessura, tipo_linha)
fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imagem, 'Deltinha', (50,50), fonte, 0.8, (255, 0,0), 2, cv2.LINE_AA)





# GERA IMAGEM ----------------------------------------------------------------------------------------------------------------------------------------------
cv2.imshow("Imagem", imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

from detectaCor import detectar_cor_especifica
from detectaForma import detectar_quadrado

# Parâmetros para cálculo da distância
DISTANCIA_FOCAL_PIXELS = 750
LARGURA_REAL_CM = 5.65  # 5,64 APROX   #Não' pode ser definido assim , alterar teste


def calcular_distancia(largura_em_pixels):
    """Calcula a distância da câmera até um objeto conhecido."""
    if largura_em_pixels == 0:
        return -1  # Evita divisão por zero

    distancia = (LARGURA_REAL_CM * DISTANCIA_FOCAL_PIXELS) / largura_em_pixels
    return distancia


# LOOP PRINCIPAL ---------------------------------------------------------------------------------------------------------------------------------------------------
camera = cv2.VideoCapture(0)

LARGURA_DESEJADA = 1280
ALTURA_DESEJADA = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_DESEJADA)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_DESEJADA)

# Verificar se a câmera aplicou as configurações
largura_real = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
altura_real = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolução da câmera definida para: {largura_real}x{altura_real}")


if not camera.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

print("Câmera iniciada. Pressione 'q' para sair.")
print("Mostre um objeto quadrado para a câmera.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Chama a função de detecção
    info_quadrado = detectar_quadrado(frame)

    # Se a função retornou informações de um quadrado
    if info_quadrado:
        contorno_detectado = info_quadrado["contour"]

        # Indentifica se a cor ta presente
        cor_alvo_encontrada = detectar_cor_especifica(
            frame, contorno_detectado)

        # Exibe a área do quadrado encontrado
        print(f"Área: {info_quadrado['area']:2f} pixels")

        distancia = calcular_distancia(info_quadrado['largura_pixels'])
        print(f"Distância: {distancia} em centímetros")

        if cor_alvo_encontrada:
            cor_borda = (0, 0, 255)  # Vermelho para o alvo correto
            texto = f"ALVO DETECTADO ({distancia:.1f} cm)"
            print("Alvo com cor específica")
        else:
            cor_borda = (0, 255, 0)  # Verde para um quadrado qualquer
            texto = "Quadrado"

        # ### CORREÇÃO 3: Usa a variável 'cor_borda' para desenhar o contorno. ###
        cv2.drawContours(frame, [contorno_detectado], -1, cor_borda, 3)

        cv2.circle(frame, info_quadrado['center'], 5, (0, 0, 255), -1)

        # ### CORREÇÃO 4: Usa a variável 'texto' para escrever na tela. ###
        texto_pos = (info_quadrado['center'][0] - 80,
                     info_quadrado['center'][1] - 20)
        cv2.putText(frame, texto, texto_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostra o resultado na janela
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('frame_unico.jpg', frame)
        print("Frame salvo!")

camera.release()
cv2.destroyAllWindows()
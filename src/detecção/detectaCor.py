import cv2
import numpy as np

# Detecta a cor #2B0000
def detectar_cor_especifica(frame, contour):
    # Converte o frame para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites (lower/upper) para a cor #2B0000 (Vermelho Escuro) em HSV ---  RGB(43, 0, 0) -> HSV(0, 255, 43)
    # Faixa 1 (vermelhos mais baixos)
    lower_red1 = np.array([0, 100, 20])
    upper_red1 = np.array([10, 255, 100])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

    # Faixa 2 (vermelhos mais altos)
    lower_red2 = np.array([170, 100, 20])
    upper_red2 = np.array([180, 255, 100])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    # Combina as duas máscaras de cor
    mask_cor = mask1 + mask2

    # Cria uma máscara para a forma do contorno detectado
    mascara_contorno = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.drawContours(mascara_contorno, [contour], -1, 255, -1)

    # Os pixels dentro da faixa de cor definida
    mascara_combinada = cv2.bitwise_and(mask_cor, mask_cor, mask=mascara_contorno)


    # Se houver pixels na máscara combinada, a cor foi encontrada dentro da forma
    if cv2.countNonZero(mascara_combinada) > 50:  # 50 pixels de tolerância
        return True
    return False


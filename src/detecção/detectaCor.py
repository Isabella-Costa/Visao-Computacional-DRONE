import cv2
import numpy as np

# Detecta a cor #2B0000
def detectar_cor_especifica(frame, contour):
    # Converte o frame para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites (lower/upper) para a cor #2B0000 (Vermelho Escuro) em HSV ---  RGB(43, 0, 0) -> HSV(0, 255, 43)
    lower_red = np.array([0, 100, 20])
    upper_red = np.array([100, 250, 100])

    # Os pixels dentro da faixa de cor definida
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Cria uma máscara para a forma do contorno detectado
    mascara_contorno = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.drawContours(mascara_contorno, [contour], -1, 255, -1)

    mascara_combinada = cv2.bitwise_and(mascara_contorno, mask)

    # Se houver pixels na máscara combinada, a cor foi encontrada dentro da forma
    if cv2.countNonZero(mascara_combinada) > 50:  # 50 pixels de tolerância
        return True
    return False


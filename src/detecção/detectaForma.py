import cv2
import numpy as np


# Parâmetros para o Detector de Bordas Canny.
# Ajustar esses valores para melhora a detecção em diferentes iluminações.
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Área mínima do contorno para ser considerado um quadrado em pixel quadrados
MIN_AREA = 300

def detectar_quadrado(frame):

    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # BUSCA DE CONTORNOS -----------------------------------------------------------------------------------------------------------------------------
    # RETR_TREE para obter a hierarquia completa.
    contours, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # A variável hierarchy terá o formato [[[]]] se não houver contornos
    if hierarchy is None:
        return None

    # LOOP PARA USAR A HIERARQUIA ----------------------------------------------------------------------------------------------------------------------------
    for i, c in enumerate(contours):

        # Cálculo da àrea
        area = cv2.contourArea(c)
        # Ignora contornos que são muito pequenos
        if area < MIN_AREA:
            continue

        # VERIFICAÇÃO DA HIERARQUIA
        # Se o contorno não tem pai (hierarchy[0][i][3] == -1).
        if hierarchy[0][i][3] == -1:
           # Analisa a Geometria do Contorno
            peri = cv2.arcLength(c, True)

            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            # Se a forma aproximada tem 4 vértices
            if len(approx) == 4:
                # Pega a caixa delimitadora para verificar a proporção
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Se a proporção for próxima de 1, confirmamos que é um quadrado
                if 0.90 <= aspect_ratio <= 1.10:
                    center_coords = (int(x + w / 2), int(y + h / 2))

                # só ta pegando o externo
                    return {"contour": c, "center": center_coords, "area": area, "largura_pixels": w}

    # Se o loop terminar e nenhum quadrado for encontrado, retorna None
    return None


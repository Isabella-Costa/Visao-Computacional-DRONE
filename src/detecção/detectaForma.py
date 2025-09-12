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

    # Busca de contornos
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Hierarquia dos contornos
    if hierarchy is None:
        return None

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        if hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if 0.90 <= aspect_ratio <= 1.10:  # é quadrado
                    center_coords = (int(x + w / 2), int(y + h / 2))
                    return {"contour": c, "center": center_coords, "area": area, "largura_pixels": w,"quadrado": True}

    return None


import cv2
import numpy as np


# Parâmetros para o Detector de Bordas Canny.
# Ajustar esses valores para melhorar a detecção em diferentes iluminações.
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Parâmetros para cálculo da distância
DISTANCIA_FOCAL_PIXELS = 750
LARGURA_REAL_CM = 13  # 5,64 APROX   #Não' pode ser definido assim , alterar teste


# Área mínima do contorno para ser considerado um quadrado.
# Aumentei um pouco para evitar ruídos maiores - quanto maior mais ruido
# em pixel quadrados
MIN_AREA = 300


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

# Detecta quadrado


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

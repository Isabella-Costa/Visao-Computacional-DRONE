import cv2
import numpy as np

# Parâmetros para o Detector de Bordas Canny.
# Ajustar esses valores para melhorar a detecção em diferentes iluminações.
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Área mínima do contorno para ser considerado um quadrado.
# Isso evita que pequenos ruídos na imagem sejam detectados como quadrados.
MIN_AREA = 10 #em pixels quadrados / 1000 ajuda a n ficar muito doida a câmera, mas se quiser testar com quadrado pequeno n dá


def detectar_quadrado(frame):
    
    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # Contornos
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se nenhum contorno for encontrado, não há o que fazer.
    if len(contours) == 0:
        return None

    # Iterar sobre todos os contornos encontrados (do maior para o menor)
    # Isso é melhor do que pegar apenas o maior, caso haja objetos maiores na cena.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        # Ignora contornos que são muito pequenos
        if cv2.contourArea(c) < MIN_AREA:
            continue

        # Analisa a Geometria do Contorno
        peri = cv2.arcLength(c, True)
        # Aproxima o contorno para uma forma com menos vértices
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Se a forma aproximada tem 4 vértices, é um candidato a quadrado
        if len(approx) == 4:
            # Pega a caixa delimitadora para verificar a proporção
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Se a proporção for próxima de 1, confirmamos que é um quadrado
            if 0.90 <= aspect_ratio <= 1.10:
                center_coords = (int(x + w / 2), int(y + h / 2))
                
                # Encontramos nosso quadrado! Retornamos as informações dele.
                return {
                    "contour": c,
                    "center": center_coords
                }
    
    # Se o loop terminar e nenhum quadrado for encontrado, retorna None
    return None

# Loop principal
camera = cv2.VideoCapture(0)
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
        # Desenha o contorno verde ao redor do quadrado
        cv2.drawContours(frame, [info_quadrado['contour']], -1, (0, 255, 0), 3)
        # Marca o centro com um círculo vermelho
        cv2.circle(frame, info_quadrado['center'], 5, (0, 0, 255), -1)
        
        # Colocamos o texto perto do centro do objeto detectado
        texto_pos = (info_quadrado['center'][0] - 50, info_quadrado['center'][1] - 20)
        cv2.putText(frame, "ALVO", texto_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra o resultado na janela
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('frame_unico.jpg', frame)
        print("Frame salvo!")

camera.release()
cv2.destroyAllWindows()

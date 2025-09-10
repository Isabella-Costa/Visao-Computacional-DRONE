import cv2
import numpy as np

# Parâmetros para o Detector de Bordas Canny.
# Ajustar esses valores para melhorar a detecção em diferentes iluminações.
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

#Parâmetros para cálculo da distância
DISTANCIA_FOCAL_PIXELS = 750
LARGURA_REAL_CM = 13   #Não pode ser definido assim , alterar teste


#  Área mínima do contorno para ser considerado um quadrado.
# Aumentei um pouco para evitar ruídos maiores - quanto maior mais ruido
# em pixel quadrados
MIN_AREA = 300

def detectar_quadrado(frame):
    
    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    #BUSCA DE CONTORNOS -----------------------------------------------------------------------------------------------------------------------------
    #RETR_TREE para obter a hierarquia completa.
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


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

        #VERIFICAÇÃO DA HIERARQUIA
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
                    
                #só ta pegando o externo
                    return {"contour": c, "center": center_coords, "area": area, "largura_pixels": w}
                
    # Se o loop terminar e nenhum quadrado for encontrado, retorna None
    return None


def calcular_distancia(largura_em_pixels):
    """Calcula a distância da câmera até um objeto conhecido."""
    if largura_em_pixels == 0:
        return -1 # Evita divisão por zero
        
    distancia = (LARGURA_REAL_CM* DISTANCIA_FOCAL_PIXELS) / largura_em_pixels
    return distancia





# Loop principal
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
        # Exibi a área do quadrado encontrado
        print(f"Quadrado detectado. Área: {info_quadrado['area']:2f} pixels")

        distancia = calcular_distancia(info_quadrado['largura_pixels'])
        print(f"Quadrado detectado. Distância: {distancia} em centímetros")

        # Desenha o contorno verde ao redor do quadrado
        cv2.drawContours(frame, [info_quadrado['contour']], -1, (0, 255, 0), 3)
        #print(peri)
        # Marca o centro com um círculo vermelho
        cv2.circle(frame, info_quadrado['center'], 5, (0, 0, 255), -1)
        
        # Colocamos o texto perto do centro do objeto detectado
        texto_pos = (info_quadrado['center'][0] - 50, info_quadrado['center'][1] - 20)
        cv2.putText(frame, "Quadrado", texto_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
import cv2

#Vídeo já existente
#cap = cv2.VideoCapture("OpenCV/img-video/video.mp4")

#WEBCAM
cap = cv2.VideoCapture(0)

#DADOS DE REDIMENSIONAMENTO
largura = 420
altura = 360

fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('OpenCV/video_teste.avi', fourcc, 20.0, (largura, altura))

#LOOP PARA LER E MOSTRAR OS FRAMES DO VÍDEO
while True: 

    # Ler o vídeo um frame (quadro) por vez
    #ret (abreviação de "return") -> True: Significa que o frame foi lido com sucesso.  False: Significa que não há mais frames para ler
    ret, video = cap.read()

    # video_redimensionado = cv2.resize(video, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    video_redimensionado = cv2.resize(video, (altura,largura), fx = 0, fy = 0, interpolation= cv2.INTER_AREA)

    hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)

    saida.write(hsv)

    cv2.imshow('Video', video_redimensionado)
    

    # .waitKey(X) pausa a execução do código por um número específico de milissegundos / controla a velocidade de reprodução
        # X = 0 - Espera infinitamente por uma tecla. Ideal para exibir uma única imagem.
        # X = 1 - reproduzir o vídeo o mais rápido que o seu computador conseguir processar, pois a pausa é mínima. 
        # Quanto maior o valor de X mais lento fica
    #& 0xFF ->operação de máscara de bits
    # operador "&" bit a bit  e  0xFF é a representação hexadecimal para o número 255
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


saida.release()
cap.release()

cv2.destroyAllWindows()
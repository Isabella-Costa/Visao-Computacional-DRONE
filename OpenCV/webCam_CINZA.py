import cv2

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

    
    videoRed = cv2.resize(video, (largura,altura), fx = 0 , fy = 0, )

    cv2.imshow("Video", video)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        break


cap.release()

cv2.destroyAllWindows()
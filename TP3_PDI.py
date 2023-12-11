import cv2
import numpy as np
for i in range(1, 5):
    nom_video = 'tirada_' + str(i) + '.mp4'
    nom_video_mod = 'tirada_modificada_' + str(i) + '.mp4'
    cap = cv2.VideoCapture(nom_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cont_frames_quieto = 0
    lista_centroides = []
    frame_actual = -1
    out = cv2.VideoWriter(nom_video_mod, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret==True:

            frame = cv2.resize(frame, dsize = None, fx= 0.9,fy=0.3)
            mascara = np.zeros_like(frame)
            centroids_list = []
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(hsv_frame, (0, 0, 0), (20, 255, 155))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

            # Realizar la operación de cierre morfológico
            imagen_cerrada = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
            contours,_ = cv2.findContours(imagen_cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                aspect_ratio = w/h
                if cv2.contourArea(contour) > 200 and cv2.contourArea(contour) < 1500 and 0.8 < aspect_ratio < 5:
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(mascara, [hull], -1, (255,255,255), thickness=cv2.FILLED)
            
            mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, 8, cv2.CV_32S)
            
            if retval == 6:
                lista_centroides.append(centroids)
                frame_actual += 1
                if len(lista_centroides) > 1:
                    if np.all(abs(lista_centroides[frame_actual] - lista_centroides[frame_actual-1]) < 10):
                        cont_frames_quieto += 1
                    else:
                        cont_frames_quieto = 0

            if cont_frames_quieto > 3:
                
                for label in range(1, retval):
                    component_mask = (labels == label).astype(np.uint8) * 255
                    contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        x,y,w,h = cv2.boundingRect(cnt)
                        roi=component_mask[y:y+h,x:x+w]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),2)
                        contador_numero = 0
                        kernel_morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        frame_threshold = cv2.inRange(hsv_frame[y:y+h, x:x+w], (0, 0, 175), (255, 60, 255))
                        frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel_morph_close)
                        contourst, _ = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.imshow("dado", frame_threshold)
                        cv2.waitKey(500)
                        
                        
                        for contourd in contourst:
                            hull = cv2.convexHull(contourd)
                            area = cv2.contourArea(hull)
                            if area > 7:
                                contador_numero +=1
                        cv2.putText(frame,f'{contador_numero}', (x,y), 1, 1, (200,0,0),2)


            cv2.imshow('imagen modificada', frame)
            frame = cv2.resize(frame, dsize=(width, height))
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()

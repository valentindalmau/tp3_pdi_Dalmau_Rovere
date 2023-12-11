import cv2
import numpy as np
for i in range(1, 5):
    nom_video_mod = 'tirada_modificada_' + str(i) + '.mp4'
    cap = cv2.VideoCapture(nom_video_mod)
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(frame, dsize = None, fx= 0.9,fy=0.3)
            cv2.imshow('imagen modificada',frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
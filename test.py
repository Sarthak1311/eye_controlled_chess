import mediapipe as mp 
import numpy as np 
import cv2

# init mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# capture video 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret , frame  = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1) #Mirror effect
    h,w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # becaue mediapipe required rgb

    # detect face landmarks 
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # fetching eye landmarks coordinate
            left_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]] # left eye indices  
            right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]] # right eye indices 

            def get_eye_center(eye):
                x = int(np.mean([p.x for p in eye]) * w)
                y = int(np.mean([p.y for p in eye]) * h)
                return (x, y)
            
            left_eye_center = get_eye_center(left_eye)
            right_eye_center = get_eye_center(right_eye)

            # draw eye center 
            cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)
        
        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
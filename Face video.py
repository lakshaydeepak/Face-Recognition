import cv2

face_cascade = cv2.CascadeClassifier('E:\\A\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)   # For the access of camera 0-for same device  1-for other device

while True:
    ret, frame = cap.read() # FOr capturing frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # for converting the picture into gray scale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
    # Displaying the resultant frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
    
    
import cv2

count = 0
face_cascade = cv2.CascadeClassifier('E:\\A\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)                               # For the access of camera 0-for same device  1-for other device


while True:
    count = count+1
    ret, frame = cap.read()                             # For capturing frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # for converting the picture into gray scale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        
        file_name_path = 'F:\\Git Projects\\Face-Recognition\\face dataset\\user'+str(count)+'.jpg' # For reading only the reasion og interest
        cv2.imwrite(file_name_path, roi_gray)
        
        color = (0,0,255)                               # Color of frame
        stroke = 2                                      # thickness of frame
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke) #for drawing the rectangle
        cv2.imshow('roi_gray',faces)
    
    # Displaying the resultant frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q') or count==50:
        break

cap.release()
cv2.destroyAllWindows()    
    
    
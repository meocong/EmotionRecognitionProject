import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


cap = cv2.VideoCapture('input.mp4')

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        if (w*h > 20000 or w*h < 100):
            continue
        if (w*h > 1000):
            x += w/4
            y += h/4
            w /= 2
            h /= 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y+int(h*2/3):y + h, x:x + w]
        # roi_color = img[y+int(h*2/3):y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # # for (ex, ey, ew, eh) in eyes:
        # #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
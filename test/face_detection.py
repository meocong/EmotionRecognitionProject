import numpy as np
import cv2
from keras.models import model_from_json

# from faceExpress import model_run
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_cascade3 = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

def detectFaces(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=3, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    faces3 = face_cascade3.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(3, 3),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    res = []
    for (x, y, w, h) in faces:
        # if (w * h > 20000 or w * h < 100):
        #     continue
        # if (w * h > 8000):
        #     x += w / 4
        #     y += h / 4
        #     w /= 2
        #     h /= 2
        res.append([x,y,w,h])

    epsilon = 100
    for (x, y, w, h) in faces2:
        # if (w * h > 20000 or w * h < 100):
        #     continue
        # if (w * h > 8000):
        #     x += w / 4
        #     y += h / 4
        #     w /= 2
        #     h /= 2

        kt = True
        for rec in res:
            if (rec[0] + rec[2]/2 - x - w/2)**2 + (rec[1] + rec[3]/2 - y - h/2)**2 < epsilon:
                kt = False
        if (kt == True):
            res.append([x,y,w,h])

    epsilon = 100
    for (x, y, w, h) in faces3:
        # if (w * h > 20000 or w * h < 100):
        #     continue
        # if (w * h > 8000):
        #     x += w / 4
        #     y += h / 4
        #     w /= 2
        #     h /= 2

        kt = True
        for rec in res:
            if (rec[0] + rec[2] / 2 - x - w / 2) ** 2 + (rec[1] + rec[3] / 2 - y - h / 2) ** 2 < epsilon:
                kt = False
        if (kt == True):
            res.append([x, y, w, h])

    return res

def predict_emotion(face_image_gray): # a single cropped face
    if (face_image_gray.shape[0] == 0 or face_image_gray.shape[1] == 0):
        return [0,0,0,0,0,0.1]
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

def detectEmotion(angry, fear, happy, sad, surprise, neutral):
    A = [angry, fear, happy, sad, surprise, neutral]
    x = np.max(A)
    if (x - np.min(A) < 0.1):
        return "neutral"
    if (angry == x):
        return "angry"
    if (fear == x):
        return "fear"
    if (happy == x):
        return "happy"
    if (sad == x):
        return "sad"
    if (surprise == x):
        return "surprise"
    return "neutral"


def detectListFaces(videoPath, isPrint):
    cap = cv2.VideoCapture(videoPath)
    student_list = {}
    n_student = 0
    n_checked = 0
    list_faces = {}
    counting_frame = 0
    epsilon_face = 5

    while (True):
        # Capture frame-by-frame
        ret, img = cap.read()
        if (ret == False):
            break
        counting_frame += 1
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # Our operations on the frame come here

        faces = detectFaces(img)
        check = np.zeros(len(faces))
        will_print = np.zeros(len(faces))

        epsilon = 100
        epsilon_coo = 0
        for index in student_list.keys():
            student = student_list[index]
            center = student['coo']
            counting = student['count']
            counting_num = student['counting_num']
            this_counting = 0

            for i, face in enumerate(faces):
                if (center[0] - face[0] - face[2] / 2) ** 2 + (center[1] - face[1] - face[3] / 2) ** 2 < epsilon:
                    # The same student
                    student['coo'] = [face[0] + face[2] / 2, face[1] + face[3] / 2]
                    student['counting_num'] += 1
                    this_counting += 1
                    check[i] = index
                    list_faces[index][counting_frame] = face
                elif (center[0] >= face[0] - epsilon_coo and center[1] >= face[1] - epsilon_coo and
                              center[0] <= face[0] + epsilon_coo + face[2] and center[1] <= face[1] + epsilon_coo +
                    face[3]):
                    will_print[i] = 1

            student['count'].append(this_counting)

        for i, face in enumerate(faces):
            if (will_print[i] == 1):
                continue
            x = face[0] - epsilon_face
            y = face[1] - epsilon_face
            w = face[2] + 2 * epsilon_face
            h = face[3] + 2 * epsilon_face
            face_image_gray = img_gray[y:y + h, x:x + w]
            angry, fear, happy, sad, surprise, neutral = predict_emotion(face_image_gray)
            label = detectEmotion(angry, fear, happy, sad, surprise, neutral)

            if (isPrint == True):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)

            if (check[i] > 0):
                if (isPrint == True):
                    cv2.putText(img, str(int(check[i])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # cv2.putText(img, label, (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 0)
                    continue
            else:
                n_checked += 1
                list_faces[n_checked] = {counting_frame:[x,y,w,h]}
                student_list[n_checked] = {"coo": [x + w / 2, y + h / 2], "count": [1], 'counting_num': 1}

        # Update student
        list_pop = []
        for index in student_list.keys():
            student = student_list[index]
            center = student['coo']
            counting = student['count']
            counting_num = student['counting_num']

            if (len(counting) > 60):
                counting_num -= counting[0]
                counting.pop(0)

            if (counting_num <= 0):
                # student_list.pop(index, None)
                list_pop.append(index)
                n_student -= 1
            else:
                student['count'] = counting
        for index in list_pop:
            student_list.pop(index, None)
        # Display the resulting frame
        if (isPrint == True):
            cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return list_faces

emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
detectListFaces('crop.mp4', True)

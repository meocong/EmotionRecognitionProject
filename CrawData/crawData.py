import numpy as np
import cv2
import pickle
from keras.models import model_from_json
from faceExpress.model_run import predict

# from faceExpress import model_run
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_profileface.xml')
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

emotions_lists = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def putEmoji(img, x, y, w, h, label):
    if (w == 0 or h==0 or x<=0 or y<=0):
        return img
    path = "icon/" + emotions_lists[label] + ".png"
    mask = cv2.imread(path)
    mask = cv2.resize(mask, (w,h))
    n = mask.shape[0]
    m = mask.shape[1]
    if (y+n <= img.shape[0] and x+m <= img.shape[1]):
        for c in range(0,3):
            img[y:y+n,x:x+m,c] = mask[:,:,c] * (mask[:,:,2]/255.0) +  img[y:y+n, x:x+m, c] * (1.0 - mask[:,:,2]/255.0)
    return img

def detectFaces(gray):
    n = gray.shape[0]
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=10, minSize=(20, 20),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    faces3 = face_cascade3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20),
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
        if (y+h > 9*n/10 or y+h < n/10):
            continue
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
            if (y + h > 9 * n / 10 or y + h < n / 10):
                continue
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
            if (y + h > 9 * n / 10 or y + h < n / 10):
                continue
            res.append([x, y, w, h])

    return res

def predict_emotion(face_image_gray): # a single cropped face
    if (face_image_gray.shape[0] == 0 or face_image_gray.shape[1] == 0):
        return 6
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = np.array(resized_img.reshape(1, 48, 48, 1))
    # image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = predict(image)
    return list_of_list[0]
    print(list_of_list)
    # if (list_of_list == None):
    #     return [0,0,0,0,0,0,0.1]
    angry, Disgust, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, Disgust, fear, happy, sad, surprise, neutral]

def detectEmotion(angry, Disgust, fear, happy, sad, surprise, neutral):
    A = [angry, Disgust, fear, happy, sad, surprise, neutral]
    x = np.max(A)
    if (x - np.min(A) < 0.1):
        return "neutral"
    if (Disgust == x):
        return "disgust"
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


def get_fps(cap):
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): \
            {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS): \
              {0}".format(fps))
    return fps


def detectListFaces(videoPath, isPrint):
    cap = cv2.VideoCapture(videoPath)
    student_list = {}
    n_student = 0
    n_checked = 0
    list_faces = {}
    counting_frame = 0
    counting_frame = int(counting_frame)
    counting_frame_dat = 0
    sec_count = 0
    epsilon_face = 2

    fps = get_fps(cap)
    while (True):
        # Capture frame-by-frame
        ret, img = cap.read()
        if (ret == False):
            break
        img = cv2.resize(img, (640, 480))
        counting_frame += 1
        counting_frame_dat += 1
        counting_frame = int(counting_frame)

        if (counting_frame_dat > fps):
            counting_frame_dat = counting_frame_dat % fps
            print(sec_count)
            sec_count += 1
        if (int(counting_frame) % 5 != 0):
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # Our operations on the frame come here

        faces = detectFaces(img)
        check = np.zeros(len(faces))
        will_print = np.zeros(len(faces))

        epsilon = 150
        epsilon_coo = 2
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
                    list_faces[index][counting_frame] = {'position':face,'label':6}
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
            label = predict_emotion(face_image_gray)
            text_label = emotions_lists[label]
            # label = detectEmotion(angry, Disgust, fear, happy, sad, surprise, neutral)
            if (isPrint == True):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # print(label)
            if (check[i] > 0):
                if (isPrint == True):
                    # cv2.putText(img, str(int(check[i])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 0)
                    # cv2.putText(img, str(text_label), (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 0)
                    img = putEmoji(img, x, y, w, h, label)
            else:
                n_checked += 1
                list_faces[n_checked] = {counting_frame:{'position':[x,y,w,h],'label':label}}
                # list_faces dict : id : int id-> {n_frame:[x,y,w,h] coordinate} dict

                # student id
                student_list[n_checked] = {"coo": [x + w / 2, y + h / 2], "count": [1], 'counting_num': 1}
                # student_list dict: id : int id-> {"coo" : center point , count : list , 'counting_num" = np.num(count)}
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
    pickle.dump(list_faces, open("list_faces", "wb"))
    return list_faces

detectListFaces('lesson.mp4', True)

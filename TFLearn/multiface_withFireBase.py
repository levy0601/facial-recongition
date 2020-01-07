import base64

import cv2
import face_recognition
import json
import sys
import io
import time


import numpy
import numpy as np
import requests
from PIL import Image
from model import EMR

# Database info, **TODO** Update the information below to point to your database.
URL = "https://peaceful-stock-259601.firebaseio.com//"
# For anonymous sign in, **TODO** Change the key below to be the API key of your Firebase project (Project Settings > Web API Key).
AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyDKUFVcdDaZ2XIvLUtfZhmHWyBY6y1-GTE";
headers = {'Content-type': 'application/json'}
auth_req_params = {"returnSecureToken":"true"}

# Start connection to Firebase and get anonymous authentication
connection = requests.Session()
connection.headers.update(headers)
auth_request = connection.post(url=AUTH_URL, params=auth_req_params)
auth_info = auth_request.json()
auth_params = {'auth': auth_info["idToken"]}
print(auth_info)

def url_to_image(url):
    # imgstr = re.search(r'base64,(.*)', url).group(1)
    imgstr = url
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    image = numpy.array(im)
    return image


# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
#------------------------------
obama_image = face_recognition.load_image_file("../knownFaces/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("../knownFaces/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
levy_image = face_recognition.load_image_file("../knownFaces/levy.jpg")
levy_face_encoding = face_recognition.face_encodings(levy_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    levy_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "levy"
]
#------------------------------

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Initialize object of EMR class
network = EMR()
network.build_network()

# In case you want to detect emotions on a video, provide the video file path instead of 0 for VideoCapture.
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []

# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread(emotion))

while True:
    # Again find haar cascade to draw bounding box around face

    # ret, frame = cap.read()
    # if not ret:
    #     break

    # Sending get request and obtaining the image
    get_request = connection.get(url=URL + "image.json")
    # Extracting data in json format, this is a string representing the image
    image_str = get_request.json()
    print(image_str)

    # Setting up face detection, **TODO** you need to place the haarcascase XML file onto your drive
    face_cascade = cv2.CascadeClassifier('/content/drive/My Drive/Lab-9/haarcascade_frontalface_default.xml')

    # Convert the image string into an actual image so we can process it
    frame = url_to_image(str(image_str))
    print(type(frame))

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)

    # ------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb_frame = rgb[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    results = [];
    # ------------------------------

    if len(faces) > 0:
        # draw box around faces
        # ------------------------------

        # ------------------------------

        for face,face_encoding in zip(faces,face_encodings):
            (x,y,w,h) = face
            # ------------------------------
            # results.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
            # ------------------------------
# ------------------------------
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # results.extend({'name': name})
                results.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'name': name})
                print(results)
            # ------------------------------

            frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2)
            newimg = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
            result = network.predict(newimg)
            if result is not None:
                maxindex = np.argmax(result[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,name + " : " + EMOTIONS[maxindex],(x+5,y-35), font,0.5,(255,255,255),2,cv2.LINE_AA)

    # ------------------------------
    print("Found " + str(len(results)) + " faces.")
    # Jasonify the results before sending
    data_json = json.dumps(results)
    # The URL for the part of the database we will put the detection results
    detection_url = URL + "detection.json"
    # Post the data to the database
    post_request = connection.put(url=detection_url,
                                  data=data_json, params=auth_params)
    # Make sure data is successfully sent
    print("Detection data sent: " + str(post_request.ok))
    time.sleep(1)
    # ------------------------------


    # cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
import cv2
import face_recognition
import sys
import numpy as np
from PIL import Image
from model import EMR

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
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)

    # ------------------------------

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # ------------------------------

    if len(faces) > 0:
        # draw box around faces
        for face,face_encoding in zip(faces,face_encodings):
            (x,y,w,h) = face

# ------------------------------
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
 # ------------------------------

            frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2)
            newimg = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
            result = network.predict(newimg)
            if result is not None:
                maxindex = np.argmax(result[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,name + " : " + EMOTIONS[maxindex],(x+5,y-35), font,0.5,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
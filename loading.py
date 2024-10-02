import pandas as pd
import face_recognition as fr
import pickle

data = {}
f1 = fr.load_image_file("faces/2022pecai278.jpg")
f2 = fr.load_image_file("faces/2022pecai282.jpg")
f3 = fr.load_image_file("faces/2022pecai263.jpg")

data['2022pecai278'] = fr.face_encodings(f1)[0]
data['2022pecai282'] = fr.face_encodings(f2)[0]
data['2022pecai263'] = fr.face_encodings(f3)[0]

with open("data/encodings_index.dat", 'wb') as f:
    pickle.dump(data, f)

import pandas as pd
import face_recognition as fr
import pickle

data = {}
f1 = fr.load_image_file("faces/2022pecai278.jpg")

data['2022pecai278'] = fr.face_encodings(f1)[0]

with open("data/encodings_index.dat", 'wb') as f:
    pickle.dump(data, f)

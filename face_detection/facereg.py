import numpy as np
import face_recognition as fr
import cv2 as cv

class face_recognition():
    #variable declearation
    cap = cv.VideoCapture(0)
    person_name = ""
    person_face_encodings = []
    
    def __init__(self, person):
        #extracting person name to be matched
        self.person_name = person.split("/")[-1]
        self.person_name = person_name.split(".")[0]

        #encoding person's face
        person_face = fr.load_image_file(person)
        person_face_locations = fr.face_locations(person_face)
        self.person_face_encodings = fr.face_encodings(person_face,person_face_locations)[0]

    def find_match(self, unknown):
        #encoding the captured face
        unknown_face = fr.load_image_file(unknown)
        unknown_face_locations = fr.face_locations(unknown_face)
        unknown_face_encodings = fr.face_encodings(unknown_face,unknown_face_locations)[0]

        #comparing the known and the unknown
        match_status = fr.face_compare([person_face_encodings],unknown_face_encodings)

        if match_status[0] == True:
            return 1,self.person_name
        else:
            return 0,"unknown"

    def recognize(self):
        pass

import face_recognition as fr
import cv2 as cv
import pandas as pd
import numpy as np
import pickle

class FaceRecognition():
    face_image = 0
    encodings_index ={}
    main_database = 0
    recently_matched = 0
    roll_no_package = []
    encodings_package = []
    cap = 0

    #To initialize datasets
    def __init__(self):
        self.main_database = pd.read_csv('data/main_database.csv')
        self.recently_matched = pd.read_csv('data/recently_matched.csv')
        with open("data/encodings_index.dat", 'rb') as f:
            self.encodings_index = pickle.load(f)
            self.roll_no_package = list(self.encodings_index.keys())
            self.encodings_package = list(self.encodings_index.values())
        self.cap = cv.VideoCapture(0)
        print(self.roll_no_package)
        print(self.encodings_package)
    
    #to capture face
    def capture_image(self):
        while True:
            ret, image = self.cap.read()
            if ret == True:
                cv.imshow("faceRecognition", image)
                if cv.waitKey(46) == ord('q'):
                    self.face_image = image
                    cv.destroyAllWindows()
                    break
    
            else:
                print("cant capture image")
                return
    
    #To encode image
    def encode_image(self):
        self.face_encodings = fr.face_encodings(self.face_image)[0]
    
    def add_person(self):
        print("This person seems to be not found in our database !")
        flag = str(input("Do you want to add the person [y/n]: "))
        if flag.lower() == 'y':
            lis = {}
            lis['roll_no'] =str(input("enter rollno: "))
            lis['name'] =str(input("enter name: "))
            lis['dob'] =str(input("enter dob: "))
            lis['class'] =str(input("enter class: "))
            lis['reg_no'] =int(input("enter regno: "))
            lis['gender'] =str(input("enter gender: "))
            lis['image_path'] = f"faces/{lis['roll_no']}.jpg" 
            lis['encodings_status'] = False
            
            if lis['roll_no'] not in self.main_database['roll_no']:
                self.main_database = self.main_database._append(lis, ignore_index =True)
                self.main_database.to_csv("data/main_database.csv", index = False) #writing main datas

            
            self.encodings_index[lis['roll_no']] = fr.face_encodings(self.face_image)[0]
            with open("data/encodings_index.dat",'wb') as f:
                pickle.dump(self.encodings_index,f)

            #writing image in face
            cv.imwrite(f"faces/{lis['roll_no']}.jpg", self.face_image)
        
        elif flag.lower() == 'n':
            print("ok!")

    #To compare the known_face with an unknown_face
    def compare(self):
        try:            
            face_locations = fr.face_locations(self.face_image)
            if len(face_locations) == 1:
                face_encodings = fr.face_encodings(self.face_image)[0]
            else:
                print("more than one face or no face found!")
                return '0'
        except:
            print("can't compare face!")
            return '0'

        #check for match in recently matched database
        for known_encodings,roll_no in zip(self.encodings_package,self.roll_no_package):
            status = fr.compare_faces([known_encodings], face_encodings)[0]

            if status == True:
                return roll_no
            
        self.add_person()
        return '0'

    #return data of the recognized face   
    def return_details(self, roll_no):
        print(f"data for the rollno : {roll_no}")
        data = self.main_database[self.main_database['roll_no']==roll_no]
        print(data)
        data = list(data.iloc[0, :])
        return data
    
    #destroying the initialization
    def __dest__(self):
        cv.destroyAllWindows()
        self.cap.release()

    #To execute in two ways
    #1. "recognize me" if dont have any image to compare.
    def recognize_me(self):
        self.capture_image()
        roll_no = self.compare()
        print(roll_no)
        if roll_no != '0':
            data = self.return_details(roll_no)
            return data
        else:
            print("thank you")
            return False

    #2. "recognize" if the user have image
    def recognize(self, image_path):
        self.face_image = fr.load_image_file(image_path)
        roll_no = self.compare()
        if roll_no != '0':
            data = self.return_details(roll_no)
            return data
        else:
            print("thank you")
            return False
def main():
    student = FaceRecognition()
    flag = str(input("Do you have image [y/n]: "))
    
    if flag.lower() == 'y':
        image_path = str(input("enter the image path (eg. face/image.jpg): "))
        data = student.recognize(image_path)
        
        if data:
            #printing datas
            for i in range(len(data)-1):
                print(f"{i+1} :",data[i])
    
    elif flag.lower() == 'n':
        data = student.recognize_me()
        
        if data:
            #printing datas
            for i in range(len(data)-1):
                print(f"{i+1} :",data[i])
    
    else:
        print("Query cant understandable sorry! ,")
        flag = input("Do yo wnat to continue or exit [y/n]:" )
        if flag.lower() == 'y':
            main()
        else:
            print("thank you")

main()


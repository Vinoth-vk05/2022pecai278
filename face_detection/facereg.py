from PIL import Image
import numpy as np
import face_recognition as fr
import cv2 

image_path = "faces/group_photo.jpeg"
image = fr.load_image_file(image_path)
face_locations = fr.face_locations(image)

#face count
count = 1

for face_location in face_locations:
    print("face_found : ",end="")
    top, right, bottom, left = face_location

    img = cv2.imread(image_path)
    img = img[top:bottom, left:right]

    cv2.imwrite(f"faces/face{count}.jpg", img)
    print(f"face{count} saved")
    
    cv2.imshow(f"face{count}",img)

    #count incrementation
    count += 1

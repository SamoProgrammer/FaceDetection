import cv2
import numpy as np
import face_recognition
import os
imagesPath = "StudentsImages"
studentImages = []
studentNames = []
studentImagesFileList = os.listdir(imagesPath)
for image in studentImagesFileList:
    currentImage = cv2.imread(f'{imagesPath}/{image}')
    studentImages.append(currentImage)
    studentNames.append(os.path.splitext(image)[0])
# print(studentNames)


def getImagesEncodings(images):
    EncodingsList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageEncode = face_recognition.face_encodings(image)[0]
        EncodingsList.append(imageEncode)
    return EncodingsList


studentImagesEncodings = getImagesEncodings(studentImages)
webcamFrame = cv2.VideoCapture(0)
while True:
    tempFrame, tempSuccess = webcamFrame.read()
    resizedTempFrame = cv2.resize(tempFrame, (0, 0), None, 0.25, 0.25)
    resizedTempFrame = cv2.cvtColor(resizedTempFrame, cv2.COLOR_BGR2RGB)
    currentFacesLocation = face_recognition.face_locations(resizedTempFrame)
    tempFrameEncodings = face_recognition.face_encodings(
        resizedTempFrame, currentFacesLocation)[0]
    for frameFaceEncode, frameFaceLocation in zip(tempFrameEncodings, currentFacesLocation):
        faceCompare = face_recognition.compare_faces(
            studentImages, frameFaceEncode)
        faceCompareDistance = face_recognition.face_distance(
            studentImages, frameFaceEncode)
        faceBestMatch = np.argmin(faceCompareDistance)

        if faceCompare[faceBestMatch]:
            studentName = studentNames[faceBestMatch]
            print(studentName)
            x1, x2, y1, y2 = frameFaceLocation
            x1, x2, y1, y2 = x1*4, x2*4, y1*4, y2*4
            cv2.rectangle(tempFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(tempFrame, (x1, y2-35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(tempFrame, studentName, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    cv2.imshow('Webcam', tempFrame)
    cv2.waitKey(1)

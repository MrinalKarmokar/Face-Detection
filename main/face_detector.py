import sys
from random import randrange

import cv2

class FaceDetection:
    """ Face Detection using haarcascade from OpenCV """

    def __init__(self):
        self.trained_face_data = cv2.CascadeClassifier(
            'trained_data/haarcascade_frontalface_default.xml')
        # self.img = cv2.imread('testing_Data/mrinal.jpg')
        self.img = cv2.imread('testing_Data/mrinal_sawali.jpg')
    
    def convert_to_gray(self, arg_img):
        """ Convert Image/Frame to Grayscale """
        grayscaled_img = cv2.cvtColor(arg_img, cv2.COLOR_BGR2GRAY)
        return grayscaled_img
    
    def resize_img(self, arg_img):
        """ Resize image to fit inside screen """
        resized_img = cv2.resize(arg_img, (510, 720))
        return resized_img
    
    def face_coordinates(self, arg_img):
        """ Gives co-ordinates of faces found """
        face_coordinates = self.trained_face_data.detectMultiScale(arg_img)
        return face_coordinates

    def img_show(self, arg_img):
        """ Show Image """
        cv2.imshow("Image", arg_img)
        cv2.waitKey()
        print("Image Shown")
    
    def frame_show(self, arg_img):
        """ Show Video Frame  """
        cv2.imshow("Image", arg_img)
        cv2.waitKey(5)

    def detect_face(self):
        """ Detect single face """
        resized_image = self.resize_img(self.img)
        gray_resized_img = self.convert_to_gray(resized_image)
        face_coordinates = self.face_coordinates(gray_resized_img)
        print(f"Face Co-ordinates: {face_coordinates[0]}")
        (_x, _y, _w, _h) = face_coordinates[0]
        cv2.rectangle(resized_image, (_x, _y),
            (_x+_w, _y+_h), (randrange(255), randrange(255), randrange(255)), 3)
        try:
            self.img_show(resized_image)
        except KeyboardInterrupt:
            print("Image shown")
            cv2.destroyAllWindows()
            sys.exit()

    def detect_multi_face(self):
        """ Detect single face """
        resized_image = self.resize_img(self.img)
        gray_resized_img = self.convert_to_gray(resized_image)
        face_coordinates = self.face_coordinates(gray_resized_img)
        for i, face_coordinate in enumerate(face_coordinates):
            if i > 5:
                break
            print(f"Face Co-ordinates #{i+1}: {face_coordinate}")
            (_x, _y, _w, _h) = face_coordinate
            cv2.rectangle(resized_image, (_x, _y),
                (_x+_w, _y+_h), (randrange(255), randrange(255), randrange(255)), 3)
        try:
            self.img_show(resized_image)
        except KeyboardInterrupt:
            print("Image shown")
            cv2.destroyAllWindows()
            sys.exit()
    
    def detect_multi_face_webcam(self):
        """ Using webcam to detect face in realtime """
        try:
            webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            (_b, _g, _r) = (randrange(255), randrange(255), randrange(255))
            print('Press "CTRL+C" to stop video feed')
            while True:
                
                successful_frame_read, frame = webcam.read()
                if not successful_frame_read:
                    print(">> Unsuccessfull Frame Read")
                    break
                gray_img = self.convert_to_gray(frame)
                face_coordinates = self.face_coordinates(gray_img)
                for i, face_coordinate in enumerate(face_coordinates):
                    if i > 5:
                        break
                    # print(f"Face Co-ordinates #{i+1}: {face_coordinate}")
                    (_x, _y, _w, _h) = face_coordinate
                    cv2.rectangle(frame, (_x, _y),
                        (_x+_w, _y+_h), (_b, _g, _r), 3)
                
                self.frame_show(frame)
        except KeyboardInterrupt:
            print("Video feed stopped")
            cv2.destroyAllWindows()
            sys.exit()

    def __del__(self):
        print("Face Detection Stopped")

def main():
    """ main function """
    _fd = FaceDetection()
    _fd.detect_multi_face_webcam()

if __name__ == "__main__":
    main()

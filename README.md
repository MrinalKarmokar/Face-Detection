# Face Detection Using OpenCV

#### Haar-Cascade Data (OpenCV)
- https://github.com/opencv/opencv/tree/master/data/haarcascades
### Single Face Detect from locally saved photo
```python
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

```

https://user-images.githubusercontent.com/42796209/162168448-c81e71e5-e08a-4758-8866-2835eb07ed4b.mp4

### Multiple Face Detect from locally saved photo

```python
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
```

https://user-images.githubusercontent.com/42796209/162168594-5613b46a-6928-4a3a-9584-a3b04c4fb0b1.mp4

### Multiple Face Detect from Live Webcam feed

```python
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
```

https://user-images.githubusercontent.com/42796209/162168759-3f67f18d-2b60-4288-a873-886c1b1d2c67.mp4


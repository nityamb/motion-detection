import cv2 as cv

def get_video_capture():
    print("Enter camera source index (integer): ", end='')
    source = int(input())

    cam = cv.VideoCapture(source)
    if cam is None or not cam.isOpened():
        raise IndexError(f"Invalid camera source index provided. Cannot open camera {source}")
    return cam
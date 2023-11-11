import cv2 as cv
import yaml
from camera import get_video_capture

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)['detection']


cam = get_video_capture()
# cam =  cv.VideoCapture("cars.mp4")
print("\n\n--------Recording started... press Q to stop--------")

# Obtain first frame
read, prev_gray = cam.read()
prev_gray = cv.cvtColor(prev_gray, cv.COLOR_BGR2GRAY)
prev_gray = cv.GaussianBlur(prev_gray, (5, 5), 0)


# Read from camera
while True:
    # Obtain new frame
    read, curr_frame = cam.read()
    if not read:
        print("Camera stream cannot be read: stopped or ended?")
        break

    # cv.imshow("Vision", curr_frame)

    # Convert to grayscale and blur
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("Grayscale", curr_gray)
    curr_gray = cv.GaussianBlur(curr_gray, (5, 5), 0)
    # cv.imshow("Blur", curr_gray)

    # Find difference between previous frame and current frame 
    diff = cv.absdiff(curr_gray, prev_gray)
    # cv.imshow("Diff", diff)

    # Adjust second parameter for difference sensitivity (lower is more sensitive)
    threshold = cv.threshold(diff, config['threshold'], 255, cv.THRESH_BINARY)[1]
    cv.imshow("Threshold", threshold)
    # Dilating is an attempt to "connect" big objects into one contour instead of getting separated into several contours
    threshold = cv.dilate(threshold, None, iterations=config['dilation-iterations'])
    cv.imshow("Dilate", threshold)

    # Get contours (basically edges)
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(curr_frame, contours, 0, (0, 255, 0), 3)
    # cv.imshow("Contours", curr_frame)


    # Obtain contours (set of connected pixels with the same color) to detect separate moving objects
    for contour in contours:
        if cv.contourArea(contour) < config['contour-threshold']:
            continue
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("Vision", curr_frame)

    prev_gray = curr_gray

    # Check for request to stop recording
    if cv.waitKey(10) == ord('q'):
        print("Recording successfully stopped")
        break


cam.release()
cv.destroyAllWindows()
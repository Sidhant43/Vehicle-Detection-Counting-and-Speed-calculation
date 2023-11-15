import cv2
import numpy as np
import time

# Open the video file
cap = cv2.VideoCapture('video.mp4')
min_width_rect = 80
min_height_rect = 80
count_line_position = 550

# Initialize the background subtractor
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6  # allowable error between pixel
counter = 0

# Create a dictionary to store timestamps for vehicles
vehicle_timestamps = {}

while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 125, 4), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                # Record the timestamp when a vehicle crosses the count line
                vehicle_timestamps[counter] = time.time()
                detect.remove((x, y))
                print("Vehicle Counter: " + str(counter))

    # Calculate speed for each vehicle
    for vehicle, timestamp in vehicle_timestamps.items():
        elapsed_time = time.time() - timestamp
        if elapsed_time > 0:
            speed_mps = (min_height_rect ) / elapsed_time  # Assuming min_height_rect is in cm
            speed_kmph = speed_mps * 3.6  # Convert m/s to km/h
            print(f"Vehicle {vehicle} Speed: {speed_kmph:.2f} km/h")
        else:
            print(f"Vehicle {vehicle} Speed: Vehicle crossed the line too quickly to calculate speed.")
      
        # Place the text inside the loop where x and y are available
    cv2.putText(frame1, "Vehicle: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 245, 0), 2)

    cv2.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    if not ret:
        print("Error: Could not read the frame.")
        break

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(30) == 13:  # 30 milliseconds delay, you can adjust this value
        break

cv2.destroyAllWindows()
cap.release()


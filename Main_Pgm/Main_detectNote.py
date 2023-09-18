import cv2
from ultralytics import YOLO
import os
from paddleocr import PaddleOCR
import csv

# Suppress PaddleOCR logging
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)

# Initialization
model = YOLO('C:\\Users\\Anugrah\\Documents\\GitHub\\License-Plate-Recognition\\best_vehical.pt')
video_path = 'C:\\Users\\Anugrah\\Documents\\GitHub\\License-Plate-Recognition\\trafficCam.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)

if not os.path.exists("detected_boxes"):
    os.makedirs("detected_boxes")

ocr = PaddleOCR()


def get_next_filename(directory):
    count = len(os.listdir(directory))
    return os.path.join(directory, f"box_{count}.jpg")


with open('license_plates.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['License Plate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        print(f"Number of detections: {len(results)}")

        results = model(frame)
        color = (0, 255, 0)

        for detection in results:
            boxes_data = detection.boxes.data
            for box in boxes_data:
                x1, y1, x2, y2, confidence, _ = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

                cropped_img = frame[y1:y2, x1:x2]
                save_path = get_next_filename("detected_boxes")
                cv2.imwrite(save_path, cropped_img)

                # OCR Processing
                ocr_results = ocr.ocr(save_path)
                for line in ocr_results:
                    _, (text, _) = line
                    writer.writerow({'License Plate': text})

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import os
from paddleocr import PaddleOCR
import csv
from sort import Sort

# Suppress PaddleOCR logging
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
tracker = Sort()

processed_tracks = set()


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

        detections = model(frame).pred[0]
        detection_list = [det[:4].tolist() + [det[4]] for det in detections]

        tracks = tracker.update(detection_list)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            if track_id not in processed_tracks:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cropped_img = frame[y1:y2, x1:x2]
                save_path = get_next_filename("detected_boxes")
                cv2.imwrite(save_path, cropped_img)

                # OCR Processing
                ocr_results = ocr.ocr(save_path)
                for line in ocr_results:
                    _, (text, _) = line
                    writer.writerow({'License Plate': text})

                processed_tracks.add(track_id)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

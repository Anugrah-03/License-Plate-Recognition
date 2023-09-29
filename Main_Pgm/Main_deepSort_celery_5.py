import os
import datetime
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort
import redis
import base64
from celery import Celery
from openpyxl import Workbook
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Celery('Main_Pgm.Main_deepSort_celery_5', broker='redis://localhost:6379/0')

# Constants and Configuration
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)

r = redis.StrictRedis(host='localhost', port=6379, db=0)

def process_frame(frame, model, tracker):
    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        bbox = [int(data[i]) for i in range(4)]
        class_id = int(data[5])
        results.append([bbox, confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)
    return tracks

def initialize_video(video_path='Video_Samples/sample_6.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        exit()

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1280, 720)
    return cap

def save_detected_boxes_to_redis(image_data):
    current_time = datetime.datetime.now().strftime('%H:%M')
    key = f"image:{current_time}"
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    image_data = cv2.morphologyEx(image_data, cv2.MORPH_BLACKHAT, rectKern)
    _, buffer = cv2.imencode('.jpg', image_data)
    img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    r.set(key, img_base64)
    return key

def perform_ocr(cropped_img, ocr):
    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(cropped_img_gray, cv2.MORPH_BLACKHAT, rectKern)
    ocr_results = ocr.ocr(blackhat)
    return ocr_results, blackhat

def process_redis_keys(ocr):
    processed_data = {}
    keys = r.keys("image:*")
    logging.info(f"Processing {len(keys)} keys from Redis.")
    
    for key_to_process in keys:
        image_data = r.get(key_to_process)
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ocr_results, _ = perform_ocr(img_np, ocr)
        text_result = ocr_results[0][0][1][0]
        processed_data[key_to_process] = text_result
        r.delete(key_to_process)

    # Store in Excel if any data to process
    if processed_data:
        wb = Workbook()
        ws = wb.active
        for key, text_result in processed_data.items():
            combined_text = text_result.replace('?', '')  # Filter out any '?'
            ws.append([key.decode('utf-8'), combined_text])  

        excel_path = os.path.join(BASE_DIR, f"OCR_results.xlsx")
        wb.save(excel_path)

@app.task
def monitor_redis_keys():
    ocr = PaddleOCR()
    while True:
        if len(r.keys("image:*")) > 10:
            process_redis_keys(ocr)


def resize_bbox(bbox, reduction_percentage=0.3):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    dh = height * reduction_percentage
    y2 = int(y2 - dh)
    return [x1, y1, x2, y2]

def main():
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    model = YOLO(os.path.join(BASE_DIR, 'best_vehical.pt'))
    cap = initialize_video()
    tracker = DeepSort(max_age=20)
    ocr = PaddleOCR()

    processed_tracks = set()

    # Create directory to save detected boxes
    output_dir = os.path.join(BASE_DIR, 'detected_boxes')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_counter = 0
    while cap.isOpened():
        start = datetime.datetime.now()
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % 1 == 0:
            tracks = process_frame(frame, model, tracker)
            for track in tracks:
                if track.is_confirmed() and track.track_id not in processed_tracks:
                    ltrb = track.to_ltrb()
                    bbox = [int(val) for val in ltrb]
                    bbox_resized = resize_bbox(bbox)
                    cropped_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Save the detected box image
                    image_filename = os.path.join(output_dir, f"track_{track.track_id}_{datetime.datetime.now().strftime('%H%M%S')}.jpg")
                    cv2.imwrite(image_filename, cropped_img)

                    _ = save_detected_boxes_to_redis(cropped_img)
                    processed_tracks.add(track.track_id)

            fps_info = f"FPS: {1 / (datetime.datetime.now() - start).total_seconds():.2f}"
            cv2.putText(frame, fps_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    process_redis_keys(PaddleOCR())
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    start_time = time.time()  # <-- Add this
    
    # Start the Celery task to monitor Redis keys
    monitor_redis_keys.delay()
    
    main()
    
    end_time = time.time()  # <-- Add this
    execution_time = end_time - start_time  # <-- Add this
    print(f"Execution time: {execution_time} seconds")  # <-- Add this

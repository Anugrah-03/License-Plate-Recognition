import os
import datetime
import logging
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_RESULTS_PATH = os.path.join(BASE_DIR, 'ocr_results.txt')

# Constants and Configuration
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

def initialize_video(video_path='sample_cut.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        exit()
    
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 800, 600)
    return cap

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

def save_detected_boxes(directory, cropped_img):
    save_path = get_next_filename(directory)
    logging.info(f"Saving to: {save_path}")
    cv2.imwrite(save_path, cropped_img)
    return save_path

def perform_ocr(cropped_img, ocr):
    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, cropped_img_binary = cv2.threshold(cropped_img_gray, 128, 255, cv2.THRESH_BINARY_INV)
    ocr_results = ocr.ocr(cropped_img_binary)
    return ocr_results

def get_next_filename(directory):
    absolute_directory = os.path.join(BASE_DIR, directory)
    if not os.path.exists(absolute_directory):
        os.makedirs(absolute_directory)
    count = len(os.listdir(absolute_directory))
    return os.path.join(absolute_directory, f"box_{count}.jpg")

def main():
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    frame_counter = 0
    
    # Initialization
    model = YOLO(os.path.join(BASE_DIR, 'best_vehical.pt'))
    cap = initialize_video()
    tracker = DeepSort(max_age=20)
    ocr = PaddleOCR()
    
    processed_tracks = set()

    with open(OCR_RESULTS_PATH, 'a', encoding='utf-8') as txtfile:
        while cap.isOpened():
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % 4 != 0: 
                continue

            tracks = process_frame(frame, model, tracker)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(val) for val in ltrb]

                if track_id not in processed_tracks:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), GREEN, 2)
                    cropped_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    _ = save_detected_boxes("detected_boxes", cropped_img)

                    ocr_results = perform_ocr(cropped_img, ocr)
                    for line in ocr_results:
                        if line and line[0]:
                            _, (text, _) = line[0]
                            txtfile.write(f"{text}\n")
                    processed_tracks.add(track_id)

            end = datetime.datetime.now()
            fps_info = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

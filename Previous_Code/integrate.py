import cv2
import os
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Constants
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Functions
def is_similar(box1, box2, threshold=50):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    distance = abs((x1 + x2) / 2 - (x1_ + x2_) / 2) + abs((y1 + y2) / 2 - (y1_ + y2_) / 2)
    return distance < threshold

# Initialize video capture
video_cap = cv2.VideoCapture('sample_cut.mp4')
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (int(video_cap.get(3)), int(video_cap.get(4))))

# Initialize YOLO and DeepSort
model = YOLO("best_2.pt")
tracker = DeepSort(max_age=50)

last_saved_box = None
detected_boxes_dir = "detected_boxes"
if not os.path.exists(detected_boxes_dir):
    os.makedirs(detected_boxes_dir)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 1000, 840)

frame_no = 0

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    if not ret:
        break

    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], class_id])

    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Save image of detected number plate of the vehicle
        current_box = (xmin, ymin, xmax, ymax)
        if last_saved_box is None or not is_similar(last_saved_box, current_box, threshold=50):
            cropped_img = frame[ymin:ymax, xmin:xmax]
            save_path = os.path.join(detected_boxes_dir, f"plate_{track_id}.jpg")
            cv2.imwrite(save_path, cropped_img)
            last_saved_box = current_box

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 5)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 4)

    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    cv2.imshow("Video", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

    frame_no += 5
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    
video_cap.release()
video_writer.release()
cv2.destroyAllWindows()

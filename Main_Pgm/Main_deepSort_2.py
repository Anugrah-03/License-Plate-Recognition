'''
import sys
#sys.path.insert(0, "/home/gayathry/anaconda3/lib/python3.11/site-packages")
print(sys.path)

print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
'''

import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

#CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

video_cap = cv2.VideoCapture('License-Plate-Recognition/Video_Samples/sample_cut.mp4')
#writer = create_video_writer(video_cap, "output.mp4")

model = YOLO("License-Plate-Recognition/Main_Pgm/best_vehical.pt")
tracker = DeepSort(max_age=50)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 940, 780)

if not os.path.exists("detected_boxes"):
    os.makedirs("detected_boxes")

frame_count = 0
skip_frames = 5

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    frame_count += 1

    if not ret:
        break

    if frame_count % skip_frames == 0:

        detections = model(frame)[0]
        results = []
        
        for index, data in enumerate(detections.boxes.data.tolist()):

            '''
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            '''

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            #print(f"Bounding Box: {xmin}, {ymin}, {xmax}, {ymax}")

            class_id = int(data[5])

            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], class_id])

        tracks = tracker.update_tracks(results, frame=frame)

        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # draw the bounding box and the track id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            #cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            #cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            
            cropped_img = frame[ymin:ymax, xmin:xmax]

            if cropped_img.size == 0:
                print(f"Empty cropped image for Bounding Box: {xmin}, {ymin}, {xmax}, {ymax}")
                continue

            #save_path = os.path.join("detected_boxes", f"plate_{track_id}_{index}.jpg")
            save_path = os.path.join("detected_boxes", f"plate_{track_id}.jpg")
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, cropped_img)


        # end time to compute the fps
        end = datetime.datetime.now()
        # show the time it took to process 1 frame
        #print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        # calculate the frame per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # show the frame to our screen
        cv2.imshow("Video", frame)
        #writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
#writer.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO
import os

#print(cv2.getBuildInformation())

def get_next_index(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def is_similar(box1, box2, threshold=50):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    distance = abs((x1 + x2) / 2 - (x1_ + x2_) / 2) + abs((y1 + y2) / 2 - (y1_ + y2_) / 2)
    return distance < threshold

model = YOLO('/home/gayathry/Desktop/work/num_plate/best_2.pt')

video_path = '/home/gayathry/Desktop/work/num_plate/sample_cut.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

'''
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'MP4V' for .mp4 output
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
'''

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 940, 780)  

frame_count = 0
n = 3  
prev_frame = None
last_saved_box = None

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break
    
    if frame_count % n == 0:  
        results = model(frame)
        #print(dir(results[0]))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        font_thickness = 2
        color = (0, 255, 0)  

        for i, detection in enumerate(results):

            if not os.path.exists("detected_boxes"):
                    os.makedirs("detected_boxes")

            next_index = get_next_index("detected_boxes")

            boxes_data = detection.boxes.data

            for box in boxes_data:
                x1, y1, x2, y2, confidence, _ = map(int, box)
                current_box = (x1, y1, x2, y2)

                if last_saved_box is None or not is_similar(last_saved_box, current_box, threshold=50):
                  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

                    text = f"{confidence:.2f}"
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = x1
                    text_y = y1 - 5  
                    cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)  
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA) 

                    #print(os.getcwd())

                    cropped_img = frame[y1:y2, x1:x2] 
                    save_path = os.path.join("detected_boxes", f"box_{next_index}.jpg")
                    cv2.imwrite(save_path, cropped_img)
                    last_saved_box = current_box
                    next_index += 1

        prev_frame = frame.copy()
        cv2.imshow('Video', frame)

    else:
        if prev_frame is not None:
            cv2.imshow('Video', prev_frame)
    
    if cv2.waitKey(1):
        break

    cap.release()
    cv2.destroyAllWindows()


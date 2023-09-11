import cv2
from ultralytics import YOLO
import os

model = YOLO('best.pt')

video_path = 'sample_cut.mp4'
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Video', 940, 780)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    #print(dir(results[0]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    font_thickness = 2
    color = (0, 255, 0)  


    for i, detection in enumerate(results):
        boxes_data = detection.boxes.data
        for box in boxes_data:
            x1, y1, x2, y2, confidence, _ = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

            text = f"{confidence:.2f}"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 5  
            #cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)  
            #cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA) 

            print(os.getcwd())

            if not os.path.exists("detected_boxes"):
                os.makedirs("detected_boxes")

            cropped_img = frame[y1:y2, x1:x2]  # Crop using the bounding box coordinates
            save_path = os.path.join("detected_boxes", f"box_{i}.jpg")
            cv2.imwrite(save_path, cropped_img)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


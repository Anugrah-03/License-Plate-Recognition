import cv2
from ultralytics import YOLO
import os

def get_next_index(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

model = YOLO('/home/gayathry/Desktop/work/num_plate/best.pt')

image_path = 'Cars26_png.rf.eeea5847eab227d501e6d6966c1a6d52.jpg'
image = cv2.imread(image_path)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 940, 780)  

last_saved_box = None

results = model(image)

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
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)

        text = f"{confidence:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 5  
        #cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)  
        #cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA) 

        cropped_img = image[y1:y2, x1:x2]
        save_path = os.path.join("detected_boxes", f"box_{next_index}.jpg")
        cv2.imwrite(save_path, cropped_img)
        last_saved_box = current_box
        next_index += 1

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from paddleocr import PaddleOCR
import cv2
import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("ppocr").setLevel(logging.ERROR)
# Read image using OpenCV
img_path ='Paddle_OCR/Paddle_OCR_Testing/images/black.jpg'
image = cv2.imread(img_path)

# Create an instance of the PaddleOCR
ocr = PaddleOCR()

# Perform OCR to get both detection and recognition results
results = ocr.ocr(img_path)

# Iterate through the results and print recognized text
for line in results:
    if len(line) == 3:
        word_info, (text, _), _ = line
        print(text)

    elif len(line) == 1: 
     # In case it only returns text
        text,accuracy = line[0][1]
        print(text)

# Optional: Show the image with detected regions
#or line in results:
#    if len(line) == 3:
#        word_info, _, _ = line
#        points = word_info[0]
#        points = np.array(points).astype(int).reshape((-1, 1, 2))
#        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

#cv2.imshow('Annotated Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

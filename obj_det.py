from ultralytics import YOLO
import cv2
import os

model = YOLO("weights/yolo_best.pt")

demo_folder = "demo_images"

images = [f for f in os.listdir(demo_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
images.sort()

images_to_show = images[:5]

for img_file in images_to_show:
    img_path = os.path.join(demo_folder, img_file)
    results = model(img_path)
    
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0) 

cv2.destroyAllWindows()
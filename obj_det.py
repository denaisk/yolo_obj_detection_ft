from ultralytics import YOLO
import cv2
import os

model = YOLO("weights/yolo_best.pt")
demo_folder = "demo_images"

images = sorted([f for f in os.listdir(demo_folder) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:5]

for img_file in images:
    img_path = os.path.join(demo_folder, img_file)
    results = model(img_path)
    
    boxes = results[0].boxes
    print(f"{img_file}: найдено объектов — {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"  {label}: {conf:.2f}")

    annotated = results[0].plot()
    cv2.imshow(f"Detection — {img_file}", annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("test.jpg")

for result in results:
   print("Detected objects:")


if result.boxes is None or len(result.boxes) == 0:
    print("No objects detected.")
else:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = result.names[class_id]

        print(f"{name}: {confidence:.2f}")


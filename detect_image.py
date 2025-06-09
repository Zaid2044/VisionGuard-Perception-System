from ultralytics import YOLO
import cv2

# --- Configuration ---

MODEL_PATH = 'yolov8n.pt'
IMAGE_PATH = 'test_image.jpeg'

# --- Main Detection Logic ---
def main():
    # 1. Load the model
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # 2. Run inference on the image
    print(f"Running detection on: {IMAGE_PATH}")
    results = model.predict(source=IMAGE_PATH) 

    # 3. Process and display the results
    result = results[0]
    
    image = result.orig_img

    print("\n--- Detected Objects ---")
    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        print(f"  - Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 4. Show the final image
    cv2.imshow("YOLOv8 Detection", image)
    
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = 'test_video.mp4'

def main():
    # 1. Load the model
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # 2. Open the video file
    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
        return

    # --- Video Processing Loop ---
    print("🚀 Starting video processing... Press 'q' to quit.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Writing output to: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🏁 End of video reached.")
            break

        # Run YOLO detection
        results = model.predict(source=frame, stream=True, verbose=False)
        annotated_frame = frame

        for result in results:
            annotated_frame = result.plot()
        
        cv2.imshow("VisionGuard - Real-Time Perception", annotated_frame)
        
        out.write(annotated_frame)

        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User terminated the process.")
            break
            
    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()
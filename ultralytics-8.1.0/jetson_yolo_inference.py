"""Jetson IMX219 CSI camera + YOLOv8 realtime detection script.

Verified on: Jetson Orin Nano + IMX219 + Ultralytics 8.1.0
"""

import os
import time
import cv2
from ultralytics import YOLO

def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 640,
    display_height: int = 480,
    framerate: int = 30,
    flip_method: int = 0,
):
    """Construct GStreamer pipeline for IMX219 CSI camera."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=True sync=False"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

def main():
    # Check for display (headless mode for SSH)
    headless = not bool(os.environ.get("DISPLAY"))
    if headless:
        print("[INFO] No DISPLAY detected, running in headless mode.")

    # ========== Configuration ==========
    model_path = "train_model/best.pt"
    conf_threshold = 0.5
    iou_threshold = 0.45
    imgsz = 640

    # ========== Load Model ==========
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(model_path, task="detect")
    print(f"[INFO] Model loaded: {model_path}")

    # ========== Open CSI Camera ==========
    pipeline = gstreamer_pipeline(display_width=640, display_height=480)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[ERROR] Failed to open CSI camera")
        return

    print("[INFO] Camera ready. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)

            # Print detections
            if len(results[0].boxes) > 0:
                print(f"[DETECT] Frame {frame_count}: {len(results[0].boxes)} objects found")

            # Visualization
            annotated_frame = results[0].plot()
            
            if not headless:
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = frame_count / (time.time() - start_time)
        print(f"[INFO] Finished. Avg FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
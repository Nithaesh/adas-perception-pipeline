import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

# ==============================================================================
# 1. HARDWARE ACCELERATION SETUP
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("--- COGNITIVE PERCEPTION STACK INITIALIZING ---")
print(f"Primary hardware accelerator routed to: [{device.upper()}]")

# ==============================================================================
# 2. LOCAL FILE PATHS
# ==============================================================================
# Place your driving video file in the same directory as this script
INPUT_VIDEO = "dashcam2.mp4" 
OUTPUT_VIDEO = "predicted_output.mp4"  # The final generated video file
MODEL_WEIGHTS = "yolov8m.pt"

if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"Could not find '{MODEL_WEIGHTS}' in the local directory. Please verify the download.")
if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Missing input video stream. Please drop '{INPUT_VIDEO}' into this folder.")

# ==============================================================================
# 3. MATHEMATICAL ALIGNMENT ALGORITHMS
# ==============================================================================
def average_lane_lines(image, lines):
    left_fit, right_fit = [], []
    height = image.shape[0]
    if lines is None: 
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: 
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.3:    # Left lane
            left_fit.append((slope, intercept))
        elif slope > 0.3:   # Right lane
            right_fit.append((slope, intercept))

    lines_to_draw = []
    y1 = height                
    y2 = int(height * 0.6)     

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        x1_l = int((y1 - left_avg[1]) / left_avg[0])
        x2_l = int((y2 - left_avg[1]) / left_avg[0])
        lines_to_draw.append(((x1_l, y1, x2_l, y2), left_avg))
        
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        x1_r = int((y1 - right_avg[1]) / right_avg[0])
        x2_r = int((y2 - right_avg[1]) / right_avg[0])
        lines_to_draw.append(((x1_r, y1, x2_r, y2), right_avg))
        
    return lines_to_draw

def draw_dynamic_grid(image, averaged_lines):
    """
    Draws the perspective projection lines. If dynamic lines are lost,
    it falls back to a stationary reverse camera template so the grid never disappears.
    """
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Trackers found: Draw dynamically aligned perspective lanes
    if averaged_lines and len(averaged_lines) == 2:
        left_eq, right_eq = averaged_lines[0][1], averaged_lines[1][1]
        
        grid_levels = [
            (int(height * 0.95), "1.0m", (0, 0, 255)),   # Red Zone
            (int(height * 0.85), "3.0m", (0, 255, 255)), # Yellow Zone
            (int(height * 0.75), "5.0m", (0, 255, 0))    # Green Zone
        ]
        
        for y_pos, label, color in grid_levels:
            left_x = int((y_pos - left_eq[1]) / left_eq[0])
            right_x = int((y_pos - right_eq[1]) / right_eq[0])
            cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), color, 3)
            cv2.putText(overlay, label, (left_x - 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
        for line_coords, _ in averaged_lines:
            x1, y1, x2, y2 = line_coords
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
            
    # Trackers lost: Draw stationary backup camera guideline matrix safely
    else:
        distance_zones = [
            (int(width * 0.15), int(width * 0.85), int(height * 0.95), "1.0m", (0, 0, 255)),
            (int(width * 0.25), int(width * 0.75), int(height * 0.80), "3.0m", (0, 255, 255)),
            (int(width * 0.32), int(width * 0.68), int(height * 0.70), "5.0m", (0, 255, 0))
        ]
        # Draw side track rails
        cv2.line(overlay, (distance_zones[2][0], distance_zones[2][2]), (distance_zones[0][0], distance_zones[0][2]), (255, 255, 255), 2)
        cv2.line(overlay, (distance_zones[2][1], distance_zones[2][2]), (distance_zones[0][1], distance_zones[0][2]), (255, 255, 255), 2)
        
        # Draw cross bars
        for left_x, right_x, y_pos, label, color in distance_zones:
            cv2.line(overlay, (left_x, y_pos), (right_x, y_pos), color, 3)
            cv2.putText(overlay, label, (left_x - 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

def get_roi_mask(image):
    height, width = image.shape[:2]
    polygons = np.array([[(int(width * 0.1), height), (int(width * 0.5), int(height * 0.55)), (int(width * 0.9), height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

# ==============================================================================
# 4. INITIALIZATION ENGINE & VIDEO WRITER
# ==============================================================================
yolo_model = YOLO(MODEL_WEIGHTS)
cap = cv2.VideoCapture(INPUT_VIDEO)

# Extract original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the video encoder to save the file locally
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

FOCAL_LENGTH_PX = 500.0
REAL_HEIGHTS_METERS = {0: 1.70, 2: 1.50, 3: 1.00, 5: 2.80, 7: 3.00, 16: 0.60, 19: 1.50}

print(f"\n--- Local Headless Processing Started ---")
print(f"Encoding properties: {frame_width}x{frame_height} at {fps} FPS.")
print(f"Processing frames... Saving output directly to: {OUTPUT_VIDEO}")

# ==============================================================================
# 5. STREAM PIPELINE EXECUTION LOOP (HEADLESS)
# ==============================================================================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
        
    # --- PHASE A: LANE DETECTION & PROJECTION GRID ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    roi = get_roi_mask(canny)
    
    raw_lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
    averaged_lines = average_lane_lines(frame, raw_lines)
    processed_frame = draw_dynamic_grid(frame, averaged_lines)
    
    # --- PHASE B: OBJECT TRACKING & DEPTH MAPPING ---
    tracked_classes = list(REAL_HEIGHTS_METERS.keys())
    yolo_results = yolo_model(processed_frame, device=device, classes=tracked_classes, conf=0.45, verbose=False)[0]
    
    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_height_px = y2 - y1
        if bbox_height_px <= 0: 
            continue
            
        cls_id = int(box.cls[0])
        distance_m = (REAL_HEIGHTS_METERS.get(cls_id, 1.5) * FOCAL_LENGTH_PX) / bbox_height_px
        
        display_label = f"{yolo_model.names[cls_id]} [{distance_m:.1f}m]"
        box_color = (0, 0, 255) if distance_m < 2.0 else (0, 255, 255) if distance_m < 5.0 else (255, 0, 0)
            
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(processed_frame, display_label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
    # --- PHASE C: WRITE GENERATED FRAME TO HARD DRIVE ---
    out.write(processed_frame)
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Progress status: Processed and saved {frame_count}/{total_frames} frames...")

# Safely close files and flush hardware handles
cap.release()
out.release()

print("\n==============================================================")
print(f"SUCCESS: Predicted video generation complete!")
print(f"The final annotated file has been saved as: {OUTPUT_VIDEO}")
print("==============================================================")
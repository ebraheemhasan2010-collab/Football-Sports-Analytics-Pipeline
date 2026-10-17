import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. تحميل نموذج YOLOv8 وإعداد الفيديو
# ==========================================
model = YOLO('yolov8n.pt')

# مسار فيديو المباراة
video_path = 'match.mp4'
cap = cv2.VideoCapture(video_path)

# استخراج أبعاد الفيديو وعدد الإطارات في الثانية (FPS)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# إعداد ملف الفيديو الناتج لإخراج التحليلات البصرية
output_path = 'final_match_analytics.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# قاموس لتخزين المواقع السابقة للاعبين لحساب السرعة والمسافة
previous_positions = {}

print("⏳ جاري معالجة الفيديو ورسم لوحة البيانات والسرعات... يرجى الانتظار قليلًا.")

# ==========================================
# 2. معالجة إطارات الفيديو وتتبع اللاعبين
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تشغيل التتبع المتقدم (ByteTrack) لتثبيت هويات اللاعبين (Class 0)
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # حساب نقطة المنتصف عند قدمي اللاعب (إحداثيات الحركة)
            cx, cy = int((x1 + x2) / 2), y2
            
            # ==========================================
            # 3. خوارزمية حساب السرعة والمسافة الفورية
            # ==========================================
            speed_kmh = 0.0
            if track_id in previous_positions:
                px, py = previous_positions[track_id]
                
                # حساب المسافة بالبكسل
                pixel_dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                
                # التحويل إلى سرعة واقعية (كم/ساعة) بناءً على إطارات الفيديو
                speed_mps = pixel_dist * 0.05 * (fps / 30)
                speed_kmh = speed_mps * 3.6
                
                # ضبط سقف السرعة الحركية للإنماء والركض (35 كم/س)
                if speed_kmh > 35:
                    speed_kmh = 35.0

            previous_positions[track_id] = (cx, cy)

            # ==========================================
            # 4. رسم لوحة البيانات والمربعات على الشاشة
            # ==========================================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id} | {speed_kmh:.1f} km/h"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # حفظ الإطار المعالج
    out.write(frame)

cap.release()
out.release()
print(f"✅ تم الانتهاء بنجاح! تم حفظ الفيديو النهائي باسم: {output_path}")

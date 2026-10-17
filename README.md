# Real-Time Sports Analytics & Player Tracking Pipeline

An advanced Computer Vision pipeline designed to track players and the ball in football matches, compute tactical trajectories, and estimate real-time speeds and distances using deep learning.

## 🚀 Key Features
- **Multi-Object Tracking (MOT):** Implemented YOLOv8 and ByteTrack to assign and maintain unique persistent IDs for players across frames.
- **Kinematic Analysis:** Developed mathematical algorithms to translate 2D pixel movements into real-world physical metrics (speeds in km/h and covered distances).
- **Computer Vision Pipeline:** Built on Python, OpenCV, and Ultralytics framework, optimized for high-performance processing.

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, Ultralytics YOLOv8, OpenCV, NumPy
- **Algorithms:** ByteTrack (Object Tracking), Perspective Transformation / Homography concepts.
- **Environment:** Google Colab & Local Workstation.

## 📊 Project Output
The pipeline successfully processes match footage, overlays bounding boxes with persistent IDs, and calculates live player speeds dynamically.

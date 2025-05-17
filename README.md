# Real-time Vehicle Distance Estimation using Computer Vision

This project implements a real-time system to estimate the distance of vehicles using computer vision techniques. It combines state-of-the-art object detection and monocular depth estimation to provide accurate distance measurements from a live video feed, aimed at enhancing automotive safety and situational awareness.

## Features

- **Vehicle Detection:** Utilizes YOLOv8 to identify and localize vehicles in real-time video streams.
- **Distance Estimation:** Applies the MiDaS monocular depth model to compute depth maps and estimate distances.
- **Real-time Processing:** Designed for live feeds from a webcam, dashcam, or edge devices (e.g., Jetson Nano).
- **Alerts & Display:** Visualizes estimated distances and triggers warnings for unsafe proximity.

## Technologies Used

- **Programming Language:** Python
- **Libraries/Frameworks:** OpenCV, NumPy, TensorFlow or PyTorch
- **Models:** YOLOv8 (vehicle detection), MiDaS (depth estimation)
- **Hardware:** Standard webcam/dashcam; optionally supports edge devices like Nvidia Jetson Nano

## System Architecture

1. **Video Capture:** Acquire real-time feed from the front-facing camera.
2. **Vehicle Detection:** Detect and localize vehicles using the YOLOv8 model.
3. **Depth Estimation:** Generate a depth map for each frame using the MiDaS model.
4. **Distance Calculation:** Combine vehicle bounding boxes with depth data to estimate the distance to each vehicle.
5. **Alerts & Display:** Show computed distances on the video; trigger visual/audible warnings if a vehicle is too close.
6. **Deployment:** System can run on local computers or deploy to edge devices for real-time use.

## Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Make sure you have Python 3.7+ and pip installed.*

3. **Download Pre-trained Models**
   - YOLOv8 weights (see [YOLOv8 documentation](https://docs.ultralytics.com/))
   - MiDaS weights (see [MiDaS repository](https://github.com/intel-isl/MiDaS))

4. **Run the Application**
   ```bash
   python main.py
   ```

## Usage

- Connect your webcam or use a dashcam feed.
- The application will display the video with detected vehicles and their estimated distances.
- Warnings will be shown if any detected vehicle is within an unsafe distance.

## Demo

*Add screenshots or a link to a demo video here if available.*

## Project Structure

```
│── main.py
│── requirements.txt
│── utils/
│   ├── detection.py
│   ├── depth_estimation.py
│   └── ...
│── models/
│   ├── yolov8_weights.pt
│   └── midas_weights.pth
└── README.md
```

## Acknowledgments

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [MiDaS by Intel ISL](https://github.com/intel-isl/MiDaS)
- OpenCV, NumPy, PyTorch, TensorFlow communities

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, ideas, or bug fixes.

## License

*Specify your license here (e.g., MIT, Apache 2.0) or add a LICENSE file to the repo.*

---

**Developed by Kadapakavenu**
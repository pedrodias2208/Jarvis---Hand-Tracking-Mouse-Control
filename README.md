# Jarvis---Hand-Tracking-Mouse-Control
Jarvis - Hand-Tracking Mouse Control

A Python system that uses your webcam to track your hand in real time and control the mouse cursor — including clicking — using hand gestures only, with no physical mouse involved.

🎯 What it does
Captures live video from the webcam
Detects the 21 hand landmarks in real time using MediaPipe's HandLandmarker model
Moves the mouse cursor following the tip of the index finger
Smooths the movement with a moving average over the last frames, reducing jitter
Detects clicks via gesture: when the tip of the thumb gets close to the base of the index finger, it calculates the Euclidean distance between the two points and triggers a left click
Uses two different thresholds (hysteresis) to trigger and "reset" the click, avoiding accidental double-clicks from small hand tremors
Draws visual guides on screen (colored circles and lines) showing the tracked points and the current click state
🛠️ Technologies used
Python 3
OpenCV (cv2) — video capture, image processing, and drawing the visual guides
MediaPipe (mediapipe.tasks) — hand landmark detection via HandLandmarker
NumPy — Euclidean distance calculation between hand points
mouse — controls the operating system's cursor (movement and click)
🚀 How to run
bash
# Clone the repository
git clone https://github.com/your-username/repo-name.git
cd repo-name

# Install dependencies
pip install opencv-python mediapipe mouse numpy

# Download the hand_landmarker.task model from MediaPipe
# (available at https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
# and place it in the same folder as the script

# Run the project
python mouseMao.py

⚠️ Adjust largura_da_tela_pc and altura_da_tela_pc in the code to match your monitor's resolution.

💡 How it works (technical summary)
The webcam captures each frame, and the image is horizontally flipped for a more intuitive mirror effect
MediaPipe's HandLandmarker processes the frame in VIDEO mode and returns 21 landmarks for the detected hand
The index fingertip position (landmark 8) is used to move the cursor, mapped from the camera's resolution to the screen resolution
The last 4 positions are stored, and their average defines the final cursor position, smoothing the movement
For the click, the distance between the thumb tip (landmark 4) and the base of the index finger (landmark 5) is calculated
If the distance drops below 28 pixels, a left click is triggered; a lock (ja_executou_o_clique) prevents repeated triggering, which is only released once the distance goes back above 38 pixels — this margin between the two values prevents accidental clicks from hand jitter
📌 Possible future improvements
Add right-click and scroll gestures
Support multiple hands or switching the dominant hand
Automatic screen resolution calibration
Bundle the .task model with the repository or download it automatically on first run
📄 License

This project is licensed under the MIT License.

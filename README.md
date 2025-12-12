# Harris-corner-tracking

Harris Corner + Optical Flow Tracking

This project detects Harris corner points from an image, applies the same technique to a video, and performs optical-flow-based point tracking with real-time ROI selection.

📌 Features

Harris corner detection on a still image

Harris corner detection on every video frame

Real-time ROI selection (press S)

Optical Flow tracking using Lucas-Kanade

Colored visualizations:

Blue → Harris points

Green → Optical flow tracked points

Red → Motion vectors

Yellow → Long-term motion trails

Smooth fading effect for motion trails

Saves Harris result image as harris_image_output.jpg

📁 Files Needed

Place these files in the same folder as the script:

images.jpg — image for Harris corner detection

854982-hd_1280_720_25fps.mp4 — video to track

The Python script

🔧 Requirements

Install the required libraries:

pip install opencv-python numpy

▶️ How to Run

Put your script and media files in the same directory

Run:

python your_script_name.py


Controls:

S → Select an ROI to start tracking

Q → Quit

🎨 Output

Harris corners on image → harris_image_output.jpg

Video window showing:

Harris points

Optical flow vectors

Long-term trails

Tracked motion inside the selected ROI

🧠 How It Works
1. Harris Corner Detection

Calculated using:

cv2.cornerHarris(gray, block_size, ksize, k)


Detected points are drawn in blue.

2. Optical Flow (Lucas–Kanade)

Tracks previously detected Harris points using:

cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)


Tracked movement is visualized as:

Green points (new position)

Red lines (motion vector)

Yellow long-term trails

3. Real-Time ROI Selection

Press S → drag to select a region
Only points inside this region will be tracked.

import cv2
import numpy as np

# Files in same folder
IMAGE_PATH = "images.jpg"
VIDEO_PATH = "854982-hd_1280_720_25fps.mp4"

# Harris detector parameters
block_size = 2
ksize = 3
k = 0.04
threshold = 0.01

# Optical flow params
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def harris_points(gray):
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    pts = np.argwhere(dst > threshold * dst.max())
    pts = np.flip(pts, axis=1)  # convert y,x → x,y
    return np.float32(pts)


# -------------------------
# 1) Harris on STILL IMAGE
# -------------------------
img = cv2.imread(IMAGE_PATH)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_harris = img.copy()

pts_img = harris_points(gray_img)
for p in pts_img:
    cv2.circle(img_harris, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)  # BLUE

cv2.imwrite("harris_image_output.jpg", img_harris)
print("Saved: harris_image_output.jpg")


# -------------------------
# 2) VIDEO PROCESSING
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("Error: cannot read video")
    exit()

prev_gray = None
prev_pts = None
roi_mask = None

# Layer for long motion trails
trail_layer = np.zeros_like(frame)


while ret:
    frame_disp = frame.copy()

    # Fade old trails slightly
    trail_layer = (trail_layer * 0.97).astype(np.uint8)

    if roi_mask is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Harris points inside ROI
        harris = harris_points(gray)
        pts_roi = [p for p in harris if roi_mask[int(p[1]), int(p[0])] == 255]

        for p in pts_roi:
            cv2.circle(frame_disp, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)  # BLUE

        # Optical flow
        if prev_pts is not None and prev_gray is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )

            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]

            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()

                # persistent yellow trajectory
                cv2.line(trail_layer, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 1)

                # green point
                cv2.circle(frame_disp, (int(a), int(b)), 3, (0, 255, 0), -1)

                # red short motion vector
                cv2.line(frame_disp, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

            prev_pts = good_new.reshape(-1, 1, 2)
            if len(prev_pts) > 300:
                prev_pts = prev_pts[-300:]

        prev_gray = gray.copy()

    # overlay long yellow trails
    frame_disp = cv2.addWeighted(frame_disp, 1, trail_layer, 0.4, 0)

    # show frame
    cv2.imshow("Video - Press S to Select ROI - Q to Quit", frame_disp)
    key = cv2.waitKey(30) & 0xFF

    # real-time ROI selection
    if key == ord('s'):
        print("Select ROI...")

        roi = cv2.selectROI("Video - Press S to Select ROI - Q to Quit",
                            frame_disp,
                            showCrosshair=True)

        x, y, w, h = roi
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_mask[y:y + h, x:x + w] = 255

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        harris = harris_points(gray)

        pts_roi = [p for p in harris if roi_mask[int(p[1]), int(p[0])] == 255]
        prev_pts = np.array(pts_roi, dtype=np.float32).reshape(-1, 1, 2)
        prev_gray = gray.copy()

        # reset trails when selecting new area
        trail_layer = np.zeros_like(frame)

        print("Tracking started.")

    elif key == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

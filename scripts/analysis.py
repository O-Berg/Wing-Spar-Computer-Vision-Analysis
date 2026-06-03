import argparse
import json
import math
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from project_paths import CONFIG_DIR, DATA_DIR, FIGURES_DIR, RAW_DIR, ensure_output_dirs

matplotlib.use("TkAgg")

ensure_output_dirs()

DEFAULT_VIDEO = str(RAW_DIR / "IMG_5710.mov")
KNOWN_HEIGHT_MM = 75.0
MIN_TRACK_POINTS = 16
DEFAULT_SPEED = 1.0
DEFAULT_MAX_HEIGHT = 0
DEFAULT_TRACKER = "template"
DEFAULT_MAX_FEATURES = 100
DEFAULT_SEARCH_MARGIN = 90
DEFAULT_ROI_SCALE = 1.1
DEFAULT_TEMPLATE_ANGLE_RANGE = 10
DEFAULT_TEMPLATE_ANGLE_STEP = 0.1
DEFAULT_TEMPLATE_MIN_SCORE = 0.18
DEFAULT_TEMPLATE_MATCH_SCALE = 0.25
DEFAULT_HUE_TOLERANCE = 7
DEFAULT_SAT_TOLERANCE = 40
DEFAULT_VAL_TOLERANCE = 50
DEFAULT_LOOP = True
DEFAULT_SHOW_TRACE = True
DEFAULT_TRACE_LENGTH = 500
DEFAULT_SETUP_FILE = str(CONFIG_DIR / "analysis_setup.json")
DEFAULT_USE_SAVED_SETUP = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a video with live displacement and angle overlay."
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"Path to the video file. Default: {DEFAULT_VIDEO}",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Playback speed multiplier. Example: 2.0 plays twice as fast.",
    )
    parser.add_argument(
        "--known-height-mm",
        type=float,
        default=KNOWN_HEIGHT_MM,
        help="Known calibration distance in millimetres.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help="Maximum display height. Default: fit to the screen height.",
    )
    parser.add_argument(
        "--tracker",
        choices=("template", "hybrid", "color", "lk"),
        default=DEFAULT_TRACKER,
        help="Tracking mode. Template uses the cropped filtered ROI as the reference.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Maximum ORB features used by the hybrid tracker.",
    )
    parser.add_argument(
        "--min-track-points",
        type=int,
        default=MIN_TRACK_POINTS,
        help="Minimum matched/tracked points required to trust a feature transform.",
    )
    parser.add_argument(
        "--search-margin",
        type=int,
        default=DEFAULT_SEARCH_MARGIN,
        help="Pixel margin around the previous tracked polygon used for feature search.",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=DEFAULT_ROI_SCALE,
        help="Scale factor for the polygon-shaped search gate around the current ROI.",
    )
    parser.add_argument(
        "--template-angle-range",
        type=float,
        default=DEFAULT_TEMPLATE_ANGLE_RANGE,
        help="Angle range in degrees searched by template mode around the previous angle.",
    )
    parser.add_argument(
        "--template-angle-step",
        type=float,
        default=DEFAULT_TEMPLATE_ANGLE_STEP,
        help="Angle step in degrees for template mode.",
    )
    parser.add_argument(
        "--template-min-score",
        type=float,
        default=DEFAULT_TEMPLATE_MIN_SCORE,
        help="Minimum binary-mask overlap score accepted by template mode.",
    )
    parser.add_argument(
        "--template-match-scale",
        type=float,
        default=DEFAULT_TEMPLATE_MATCH_SCALE,
        help="Scale factor used internally for template matching. Lower is faster.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Skip color picking and track without HSV color filtering.",
    )
    parser.add_argument(
        "--hue-tolerance",
        type=int,
        default=DEFAULT_HUE_TOLERANCE,
        help="HSV hue tolerance for the picked target color.",
    )
    parser.add_argument(
        "--sat-tolerance",
        type=int,
        default=DEFAULT_SAT_TOLERANCE,
        help="HSV saturation tolerance for the picked target color.",
    )
    parser.add_argument(
        "--val-tolerance",
        type=int,
        default=DEFAULT_VAL_TOLERANCE,
        help="HSV value tolerance for the picked target color.",
    )
    parser.add_argument(
        "--hide-debug",
        action="store_true",
        help="Hide the live right-side debug view of the tracking mask.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOOP,
        help="Loop the video playback and reset tracking when the video ends.",
    )
    parser.add_argument(
        "--show-trace",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_TRACE,
        help="Draw the tracked center path over time.",
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=DEFAULT_TRACE_LENGTH,
        help="Maximum number of recent tracked center points to keep in the trace.",
    )
    parser.add_argument(
        "--setup-file",
        default=DEFAULT_SETUP_FILE,
        help="JSON file used to save/load calibration, ROI, and color setup.",
    )
    parser.add_argument(
        "--use-saved-setup",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SAVED_SETUP,
        help="Reuse the saved calibration, n-gon ROI, and color samples when possible.",
    )
    return parser.parse_args()


def video_window_flags():
    return cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO


def set_video_window(window, frame):
    height, width = frame.shape[:2]
    cv2.namedWindow(window, video_window_flags())
    cv2.resizeWindow(window, width, height)


def load_setup(path, frame_shape, known_height_mm):
    setup_path = Path(path)
    if not setup_path.exists():
        return None

    try:
        with setup_path.open("r", encoding="utf-8") as file:
            setup = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    height, width = frame_shape[:2]
    if setup.get("frame_width") != width or setup.get("frame_height") != height:
        return None
    if setup.get("known_height_mm") != known_height_mm:
        return None
    if "mm_per_px" not in setup or "roi_polygon" not in setup:
        return None

    return setup


def save_setup(path, frame_shape, known_height_mm, mm_per_px, roi_polygon, color_samples):
    height, width = frame_shape[:2]
    setup = {
        "frame_width": width,
        "frame_height": height,
        "known_height_mm": known_height_mm,
        "mm_per_px": float(mm_per_px),
        "roi_polygon": np.asarray(roi_polygon, dtype=float).tolist(),
        "color_samples": (
            np.asarray(color_samples, dtype=int).tolist()
            if color_samples is not None
            else None
        ),
    }

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(setup, file, indent=2)


def screen_height():
    try:
        root = tk.Tk()
        root.withdraw()
        height = root.winfo_screenheight()
        root.destroy()
        return height
    except tk.TclError:
        return 900


def fit_to_height(frame, target_height):
    if target_height <= 0 or frame.shape[0] <= target_height:
        return frame, 1.0

    scale = target_height / frame.shape[0]
    width = int(round(frame.shape[1] * scale))
    resized = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def read_synced_frame(cap):
    ok, frame = cap.read()
    return ok, frame, 0


def put_text(frame, text, origin, scale=0.7, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def choose_calibration_points(frame, known_mm):
    points = []
    preview = frame.copy()
    window = "Calibration - click two points that span the known 150 mm height"

    def on_mouse(event, x, y, flags, userdata):
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    set_video_window(window, frame)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        preview[:] = frame
        put_text(
            preview,
            f"Click two points spanning {known_mm:g} mm, then press Enter",
            (18, 34),
            0.65,
        )
        put_text(preview, "Press R to reset, Esc to quit", (18, 64), 0.55)

        for point in points:
            cv2.circle(preview, point, 5, (0, 255, 255), -1, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(preview, points[0], points[1], (0, 255, 255), 2, cv2.LINE_AA)
            px = math.dist(points[0], points[1])
            put_text(preview, f"Scale: {known_mm / px:.3f} mm/px", (18, 94), 0.55)

        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 2:
            break
        if key == ord("r"):
            points.clear()
        if key == 27:
            raise SystemExit("Calibration cancelled.")

    cv2.destroyWindow(window)
    pixel_distance = math.dist(points[0], points[1])
    if pixel_distance <= 0:
        raise ValueError("Calibration points must not be identical.")
    return known_mm / pixel_distance


def choose_tracking_polygon(frame):
    points = []
    preview = frame.copy()
    window = "Tracking ROI - click polygon points around the object"

    def on_mouse(event, x, y, flags, userdata):
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    set_video_window(window, frame)

    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview[:] = frame
        put_text(preview, "Click around the object to make an n-gon ROI", (18, 34), 0.62)
        put_text(
            preview,
            "Enter: accept | Right click/Backspace: undo | R: reset | Esc: quit",
            (18, 64),
            0.52,
        )

        if points:
            for point in points:
                cv2.circle(preview, point, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.polylines(
                preview,
                [np.int32(points)],
                isClosed=len(points) >= 3,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            break
        if key in (8, 127) and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key == 27:
            raise SystemExit("Tracking ROI selection cancelled.")

    cv2.destroyWindow(window)
    return np.array(points, dtype=np.float32)


def polygon_mask(shape, polygon):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(polygon)], 255)
    return mask


def color_filter_preview(frame, sample_points, hue_tol, sat_tol, val_tol):
    mask, color_hsv = color_mask_from_samples(
        frame, sample_points, hue_tol, sat_tol, val_tol
    )
    if mask is None:
        preview = np.zeros_like(frame)
        put_text(preview, "Filtered preview appears here", (18, 34), 0.62)
        return preview, None

    preview = cv2.bitwise_and(frame, frame, mask=mask)
    color_count = int(cv2.countNonZero(mask))
    put_text(preview, f"Filtered pixels: {color_count}", (18, 34), 0.62, (0, 255, 255))
    put_text(preview, f"HSV: {color_hsv}", (18, 64), 0.52, (0, 255, 255))
    return preview, color_hsv


def draw_live_debug_view(frame, tracker):
    mask = tracker.color_mask(frame)
    if mask is None:
        return np.zeros_like(frame)
    return cv2.bitwise_and(frame, frame, mask=mask)


def choose_color_samples(frame, hue_tol, sat_tol, val_tol):
    points = []
    preview = np.hstack([frame, np.zeros_like(frame)])
    window = "Color filter - click target color samples"

    def on_mouse(event, x, y, flags, userdata):
        del flags, userdata
        if x >= frame.shape[1]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    set_video_window(window, preview)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        left = frame.copy()
        right, _ = color_filter_preview(
            frame,
            np.array(points, dtype=np.int32) if points else None,
            hue_tol,
            sat_tol,
            val_tol,
        )
        put_text(left, "Click target color sample(s), then press Enter", (18, 34), 0.62)
        put_text(
            left,
            "S: skip color | Right click/Backspace: undo | R: reset | Esc: quit",
            (18, 64),
            0.52,
        )
        put_text(left, "Original", (18, frame.shape[0] - 18), 0.58, (0, 255, 255))
        put_text(right, "Filtered", (18, frame.shape[0] - 18), 0.58, (0, 255, 255))

        for point in points:
            color = frame[point[1], point[0]].tolist()
            cv2.circle(left, point, 7, color, -1, cv2.LINE_AA)
            cv2.circle(left, point, 8, (255, 255, 255), 1, cv2.LINE_AA)

        preview = np.hstack([left, right])
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and points:
            cv2.destroyWindow(window)
            return np.array(points, dtype=np.int32)
        if key == ord("s"):
            cv2.destroyWindow(window)
            return None
        if key in (8, 127) and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key == 27:
            raise SystemExit("Color selection cancelled.")


def hue_mask(hue_channel, target_hue, tolerance):
    lower = (target_hue - tolerance) % 180
    upper = (target_hue + tolerance) % 180
    if lower <= upper:
        return cv2.inRange(hue_channel, lower, upper)
    return cv2.bitwise_or(cv2.inRange(hue_channel, 0, upper), cv2.inRange(hue_channel, lower, 179))


def color_mask_from_samples(frame, sample_points, hue_tol, sat_tol, val_tol):
    if sample_points is None or len(sample_points) == 0:
        return None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples = hsv[sample_points[:, 1], sample_points[:, 0]].astype(np.int32)
    median = np.median(samples, axis=0).astype(np.int32)
    hue, sat, val = [int(value) for value in median]

    hue_part = hue_mask(hsv[:, :, 0], hue, hue_tol)
    sat_low = max(0, int(np.min(samples[:, 1])) - sat_tol)
    sat_high = min(255, int(np.max(samples[:, 1])) + sat_tol)
    val_low = max(0, int(np.min(samples[:, 2])) - val_tol)
    val_high = min(255, int(np.max(samples[:, 2])) + val_tol)
    sv_part = cv2.inRange(
        hsv[:, :, 1:],
        np.array([sat_low, val_low]),
        np.array([sat_high, val_high]),
    )
    mask = cv2.bitwise_and(hue_part, sv_part)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    color_spec = {
        "hue": hue,
        "sat_low": sat_low,
        "sat_high": sat_high,
        "val_low": val_low,
        "val_high": val_high,
    }
    return mask, color_spec


def initial_points(gray, mask):
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=260,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=mask,
    )
    if points is not None and len(points) >= MIN_TRACK_POINTS:
        return points.reshape(-1, 2).astype(np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_TRACK_POINTS:
        raise SystemExit(
            "The filtered template is almost empty. Loosen the color tolerances "
            "or pick more color samples on the target."
        )

    sample_count = min(120, len(xs))
    sample_indices = np.linspace(0, len(xs) - 1, sample_count).astype(np.int32)
    sampled = np.column_stack([xs[sample_indices], ys[sample_indices]])
    return sampled.astype(np.float32)


def rigid_transform_no_scale(start_points, current_points):
    start_center = np.mean(start_points, axis=0)
    current_center = np.mean(current_points, axis=0)
    start_centered = start_points - start_center
    current_centered = current_points - current_center

    covariance = start_centered.T @ current_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = current_center - rotation @ start_center
    transform = np.float32(
        [
            [rotation[0, 0], rotation[0, 1], translation[0]],
            [rotation[1, 0], rotation[1, 1], translation[1]],
        ]
    )
    angle_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return transform, angle_deg


def transform_from_points(start_points, current_points):
    rough_transform, inliers = cv2.estimateAffinePartial2D(
        start_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=2000,
        confidence=0.99,
    )
    if rough_transform is None or inliers is None:
        return None, None

    inlier_mask = inliers.reshape(-1) == 1
    if np.count_nonzero(inlier_mask) < MIN_TRACK_POINTS:
        return None, None

    return rigid_transform_no_scale(start_points[inlier_mask], current_points[inlier_mask])


def contour_center_angle_area(contour):
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    if area <= 0 or not moments["m00"]:
        return None, None, 0

    center = np.float32([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) >= 5:
        _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
        angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    else:
        angle = 0.0
    return center, angle, area


def unwrap_pi_angle(angle, reference):
    while angle - reference > math.pi / 2:
        angle -= math.pi
    while angle - reference < -math.pi / 2:
        angle += math.pi
    return angle


def rotate_image_and_mask(image, mask, angle_deg):
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(rotation[0, 0])
    sin_a = abs(rotation[0, 1])
    new_width = int(height * sin_a + width * cos_a)
    new_height = int(height * cos_a + width * sin_a)
    rotation[0, 2] += new_width / 2.0 - center[0]
    rotation[1, 2] += new_height / 2.0 - center[1]
    rotated_image = cv2.warpAffine(
        image,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def window_sum(integral_image, x, y, width, height):
    return (
        integral_image[y + height, x + width]
        - integral_image[y, x + width]
        - integral_image[y + height, x]
        + integral_image[y, x]
    )


def image_rotation_matrix(angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.float32([[cos_a, sin_a], [-sin_a, cos_a]])


def image_rotation_angle(rotation):
    return math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))


class HybridTracker:
    def __init__(
        self,
        first_frame,
        first_gray,
        roi_mask,
        roi_polygon,
        max_features,
        search_margin,
        roi_scale,
        template_angle_range,
        template_angle_step,
        template_min_score,
        template_match_scale,
        color_hsv,
        color_tolerances,
    ):
        self.first_gray = first_gray
        self.previous_gray = first_gray
        self.roi_polygon = roi_polygon.reshape(-1, 1, 2).astype(np.float32)
        moments = cv2.moments(roi_polygon)
        if moments["m00"]:
            self.start_center = np.float32(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
        else:
            self.start_center = np.mean(roi_polygon, axis=0).astype(np.float32)
        self.frame_shape = first_gray.shape
        self.search_margin = search_margin
        self.roi_scale = roi_scale
        self.template_angle_range = template_angle_range
        self.template_angle_step = template_angle_step
        self.template_min_score = template_min_score
        self.template_match_scale = template_match_scale
        self.previous_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        self.color_hsv = color_hsv
        self.color_tolerances = color_tolerances
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        initial_mask = roi_mask
        if color_hsv is not None:
            first_color_mask = self.color_mask(first_frame)
            initial_mask = cv2.bitwise_and(roi_mask, first_color_mask)
        self.template_mask = initial_mask
        self.template_image = cv2.bitwise_and(first_frame, first_frame, mask=initial_mask)
        x, y, w, h = cv2.boundingRect(initial_mask)
        pad = max(8, int(0.08 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(first_frame.shape[1], x + w + pad)
        y1 = min(first_frame.shape[0], y + h + pad)
        self.template_origin = np.float32([x0, y0])
        self.template_crop = self.template_image[y0:y1, x0:x1]
        self.template_mask_crop = initial_mask[y0:y1, x0:x1]
        self.template_center = self.template_origin + np.float32(
            [self.template_crop.shape[1] / 2.0, self.template_crop.shape[0] / 2.0]
        )
        self.start_points = initial_points(first_gray, initial_mask)
        self.initial_start_points = self.start_points.copy()
        self.current_points = self.start_points.copy()
        self.display_points = self.current_points.copy()
        self.target_area = cv2.countNonZero(initial_mask)
        self.start_color_angle = 0.0
        initial_contours, _ = cv2.findContours(
            initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if initial_contours:
            contour = max(initial_contours, key=cv2.contourArea)
            _, angle, area = contour_center_angle_area(contour)
            if area:
                self.target_area = area
                self.start_color_angle = angle
        self.keypoints, self.descriptors = self.orb.detectAndCompute(
            first_gray, initial_mask
        )

    def reset(self):
        self.previous_gray = self.first_gray
        self.previous_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        self.start_points = self.initial_start_points.copy()
        self.current_points = self.start_points.copy()
        self.display_points = self.current_points.copy()

    def predicted_polygon(self):
        return cv2.transform(self.roi_polygon, self.previous_transform).reshape(-1, 2)

    def scaled_predicted_polygon(self):
        predicted = self.predicted_polygon()
        center = np.mean(predicted, axis=0)
        return center + (predicted - center) * self.roi_scale

    def local_search_mask(self):
        height, width = self.frame_shape
        scaled_polygon = self.scaled_predicted_polygon()
        x, y, w, h = cv2.boundingRect(np.int32(scaled_polygon))
        margin = self.search_margin
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + w + margin)
        y1 = min(height, y + h + margin)

        mask = np.zeros(self.frame_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(scaled_polygon)], 255)
        rectangle_mask = np.zeros(self.frame_shape, dtype=np.uint8)
        rectangle_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, rectangle_mask)
        return mask

    def local_search_bounds(self):
        height, width = self.frame_shape
        scaled_polygon = self.scaled_predicted_polygon()
        x, y, w, h = cv2.boundingRect(np.int32(scaled_polygon))
        margin = self.search_margin
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + w + margin)
        y1 = min(height, y + h + margin)
        return x0, y0, x1, y1

    def color_mask(self, frame):
        if self.color_hsv is None:
            return None

        hue_tol, _, _ = self.color_tolerances
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if isinstance(self.color_hsv, dict):
            hue = self.color_hsv["hue"]
            sat_low = self.color_hsv["sat_low"]
            sat_high = self.color_hsv["sat_high"]
            val_low = self.color_hsv["val_low"]
            val_high = self.color_hsv["val_high"]
        else:
            hue, sat, val = self.color_hsv
            _, sat_tol, val_tol = self.color_tolerances
            sat_low = max(0, sat - sat_tol)
            sat_high = min(255, sat + sat_tol)
            val_low = max(0, val - val_tol)
            val_high = min(255, val + val_tol)
        hue_part = hue_mask(hsv[:, :, 0], hue, hue_tol)
        sv_part = cv2.inRange(
            hsv[:, :, 1:],
            np.array([sat_low, val_low]),
            np.array([sat_high, val_high]),
        )
        mask = cv2.bitwise_and(hue_part, sv_part)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def current_search_mask(self, frame):
        search_mask = self.local_search_mask()
        color_mask = self.color_mask(frame)
        if color_mask is None:
            return search_mask
        return cv2.bitwise_and(search_mask, color_mask)

    def filtered_frame(self, frame, mask=None):
        if mask is None:
            mask = self.color_mask(frame)
        if mask is None:
            return frame.copy()
        return cv2.bitwise_and(frame, frame, mask=mask)

    def transform_from_color_blob(self, frame):
        color_mask = self.current_search_mask(frame)
        return self.transform_from_color_mask(color_mask, use_blob_angle=True)

    def transform_from_absolute_color(self, frame):
        color_mask = self.color_mask(frame)
        if color_mask is None:
            return None, None, 0
        return self.transform_from_color_mask(color_mask, use_blob_angle=True)

    def transform_from_color_mask(self, color_mask, use_blob_angle):
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None, 0

        candidates = []
        for contour in contours:
            center, angle, area = contour_center_angle_area(contour)
            if area >= 40 and center is not None:
                area_ratio = area / max(self.target_area, 1.0)
                score = abs(math.log(max(area_ratio, 1e-6)))
                candidates.append((score, contour, center, angle, area))

        if not candidates:
            return None, None, 0

        _, _, center, blob_angle, area = min(candidates, key=lambda item: item[0])
        if area < 40:
            return None, None, 0

        if use_blob_angle:
            previous_angle = math.atan2(
                self.previous_transform[1, 0], self.previous_transform[0, 0]
            )
            angle_delta = blob_angle - self.start_color_angle
            angle_delta = unwrap_pi_angle(angle_delta, previous_angle)
            cos_a = math.cos(angle_delta)
            sin_a = math.sin(angle_delta)
            rotation = np.float32([[cos_a, -sin_a], [sin_a, cos_a]])
        else:
            rotation = self.previous_transform[:, :2]
        translation = center - rotation @ self.start_center
        transform = np.float32(
            [
                [rotation[0, 0], rotation[0, 1], translation[0]],
                [rotation[1, 0], rotation[1, 1], translation[1]],
            ]
        )
        angle_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return transform, angle_deg, int(area)

    def update_lk(self, gray):
        if len(self.current_points) < MIN_TRACK_POINTS:
            return None, None, len(self.current_points), "LK"

        next_points, lk_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.current_points.reshape(-1, 1, 2),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if next_points is not None:
            good = lk_status.reshape(-1) == 1
            self.current_points = next_points.reshape(-1, 2)[good]
            self.start_points = self.start_points[good]
            self.display_points = self.current_points.copy()
        else:
            self.previous_gray = gray
            return None, None, 0, "LK"

        if len(self.current_points) < MIN_TRACK_POINTS:
            self.previous_gray = gray
            return None, None, len(self.current_points), "LK"

        self.previous_gray = gray
        transform, angle_deg = transform_from_points(
            self.start_points, self.current_points
        )
        if transform is not None:
            self.previous_transform = transform
        return transform, angle_deg, len(self.current_points), "LK"

    def update_orb(self, frame, gray):
        if self.descriptors is None or len(self.keypoints) < MIN_TRACK_POINTS:
            return None, None, 0

        current_keypoints, current_descriptors = self.orb.detectAndCompute(
            gray, self.current_search_mask(frame)
        )
        if current_descriptors is None or len(current_keypoints) < MIN_TRACK_POINTS:
            return None, None, 0

        matches = self.matcher.match(self.descriptors, current_descriptors)
        if len(matches) < MIN_TRACK_POINTS:
            return None, None, len(matches)

        matches = sorted(matches, key=lambda match: match.distance)
        keep = matches[: min(80, len(matches))]
        start = np.float32([self.keypoints[m.queryIdx].pt for m in keep])
        current = np.float32([current_keypoints[m.trainIdx].pt for m in keep])
        transform, angle_deg = transform_from_points(start, current)
        if transform is not None:
            self.display_points = current.copy()
        return transform, angle_deg, len(keep)

    def _build_rotated_template_cache(self, previous_angle):
        """Pre-rotate the template mask for every search angle and cache results."""
        scale = min(1.0, max(0.1, self.template_match_scale))
        angle_range = max(0.0, self.template_angle_range)
        angle_step = max(0.05, self.template_angle_step)
        offsets = np.arange(-angle_range, angle_range + 0.5 * angle_step, angle_step)
        cache = []
        for offset in offsets:
            angle = previous_angle + float(offset)
            _, tmask = rotate_image_and_mask(
                self.template_crop, self.template_mask_crop, angle
            )
            if scale < 1.0:
                tmask = cv2.resize(tmask, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
            if cv2.countNonZero(tmask) < 20:
                continue
            tbin = (tmask > 0).astype(np.float32)
            cache.append((angle, tbin, tmask.shape))
        return cache, scale

    def update_template(self, frame):
        color_mask = self.color_mask(frame)
        if color_mask is None:
            color_mask = np.full(self.frame_shape, 255, dtype=np.uint8)
        x0, y0, x1, y1 = self.local_search_bounds()
        search = color_mask[y0:y1, x0:x1]
        if search.size == 0:
            return None, None, 0, "template"

        scale = min(1.0, max(0.1, self.template_match_scale))
        if scale < 1.0:
            search = cv2.resize(search, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_NEAREST)
        search_f = (search > 0).astype(np.float32)

        previous_angle = image_rotation_angle(self.previous_transform[:, :2])

        # Rebuild cache only when the centre angle has shifted by more than half a step
        angle_step = max(0.05, self.template_angle_step)
        cached_angle = getattr(self, "_cache_angle", None)
        if cached_angle is None or abs(previous_angle - cached_angle) > angle_step * 0.5:
            self._template_cache, _ = self._build_rotated_template_cache(previous_angle)
            self._cache_angle = previous_angle

        best_score = -1.0
        best = None
        for angle, tbin, tshape in self._template_cache:
            th, tw = tshape[:2]
            if th > search_f.shape[0] or tw > search_f.shape[1]:
                continue
            # TM_CCORR_NORMED gives values in [0,1] directly — no IoU loop needed
            result = cv2.matchTemplate(search_f, tbin, cv2.TM_CCORR_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best = angle, loc, tshape

        if best is None or best_score < self.template_min_score:
            return None, None, 0, "template"

        angle, location, template_shape = best
        height, width = template_shape[:2]
        matched_center = np.float32(
            [
                x0 + (location[0] + width / 2.0) / scale,
                y0 + (location[1] + height / 2.0) / scale,
            ]
        )
        rotation = image_rotation_matrix(angle)
        translation = matched_center - rotation @ self.template_center
        transform = np.float32(
            [
                [rotation[0, 0], rotation[0, 1], translation[0]],
                [rotation[1, 0], rotation[1, 1], translation[1]],
            ]
        )
        self.current_points = cv2.transform(
            self.start_points.reshape(-1, 1, 2), transform
        ).reshape(-1, 2)
        self.display_points = self.current_points.copy()
        self.previous_transform = transform
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return transform, angle, int(round(best_score * 1000)), "template"

    def update(self, frame, gray, mode):
        if mode == "template":
            return self.update_template(frame)

        if mode == "color":
            transform, angle_deg, area = self.transform_from_absolute_color(frame)
            if transform is not None:
                self.current_points = cv2.transform(
                    self.start_points.reshape(-1, 1, 2), transform
                ).reshape(-1, 2)
                self.display_points = self.current_points.copy()
                self.previous_transform = transform
                self.previous_gray = gray
                return transform, angle_deg, area, "color-global"
            return None, None, 0, "color-global"

        if mode == "hybrid":
            transform, angle_deg, match_count = self.update_orb(frame, gray)
            if transform is not None:
                self.current_points = cv2.transform(
                    self.start_points.reshape(-1, 1, 2), transform
                ).reshape(-1, 2)
                self.display_points = self.current_points.copy()
                self.previous_transform = transform
                self.previous_gray = gray
                return transform, angle_deg, match_count, "ORB"

            transform, angle_deg, area = self.transform_from_color_blob(frame)
            if transform is not None:
                self.current_points = cv2.transform(
                    self.start_points.reshape(-1, 1, 2), transform
                ).reshape(-1, 2)
                self.display_points = self.current_points.copy()
                self.previous_transform = transform
                self.previous_gray = gray
                return transform, angle_deg, area, "color"

        return self.update_lk(gray)


def draw_tracking_overlay(
    frame,
    corners,
    center,
    tracked_points,
    trace_points,
    dx_mm,
    dy_mm,
    angle_deg,
    status_lines,
):
    if corners is not None:
        cv2.polylines(
            frame,
            [np.int32(corners)],
            isClosed=True,
            color=(0, 220, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if tracked_points is not None:
        for point in np.int32(tracked_points):
            cv2.circle(frame, tuple(point), 3, (255, 80, 0), -1, cv2.LINE_AA)

    if len(trace_points) >= 2:
        trace = np.int32(trace_points).reshape(-1, 1, 2)
        cv2.polylines(
            frame,
            [trace],
            isClosed=False,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        for point in np.int32(trace_points[:: max(1, len(trace_points) // 40)]):
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1, cv2.LINE_AA)

    if center is not None:
        cx, cy = np.int32(center)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (0, 255, 0), 2, cv2.LINE_AA)

    if isinstance(status_lines, str):
        status_lines = [status_lines] if status_lines else []

    panel_h = 132 + 24 * len(status_lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (520, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    put_text(frame, "Wing spar live tracking", (16, 30), 0.75, (0, 220, 255))
    put_text(frame, f"Angle from start: {angle_deg:+.2f} deg", (16, 62), 0.62)
    put_text(frame, f"Horizontal displacement: {dx_mm:+.4f} m", (16, 90), 0.62)
    put_text(frame, f"Vertical displacement:   {dy_mm:+.4f} m", (16, 118), 0.62)

    for index, line in enumerate(status_lines):
        put_text(frame, line, (16, 150 + index * 24), 0.52, (0, 255, 255))



class PlotWorker:
    """
    Live 2x2 plot grid that redraws at ~10 Hz in a background thread.
    On quit, each subplot is exported as a separate 300 DPI image.
    Redraws by clearing and re-scattering each update — the only reliable
    approach for scatter plots whose point count grows every frame.
    """

    REDRAW_INTERVAL = 0.10

    _C_DISP_H      = "#1f6eb5"   # steel blue   — horizontal scatter
    _C_DISP_V      = "#b03a2e"   # deep crimson — vertical scatter
    _C_ANGLE       = "#7b2fbe"   # vivid purple — twist scatter
    _C_GRID        = "#efefef"
    _C_ZERO        = "#c0c0c0"
    _CMAP          = "plasma"

    def __init__(self, fps: float):
        self._fps = fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._dx: list[float] = []
        self._dy: list[float] = []
        self._angle: list[float] = []
        self._live_fig = None
        self._axes: list = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _style_ax(self, ax, title, xlabel, ylabel, fontsize_title=10):
        ax.set_facecolor("white")
        ax.set_title(title, pad=7, fontsize=fontsize_title,
                     fontweight="bold", color="#111111")
        ax.set_xlabel(xlabel, labelpad=5, fontsize=8, color="#333333")
        ax.set_ylabel(ylabel, labelpad=5, fontsize=8, color="#333333")
        ax.tick_params(axis="both", labelsize=7, colors="#444444")
        ax.grid(True, color=self._C_GRID, linewidth=0.5, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#bbbbbb")
            ax.spines[sp].set_linewidth(0.8)

    def _make_cbar(self, fig, sc, ax):
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Time  (s)", fontsize=7, color="#444444")
        cbar.ax.tick_params(labelsize=6, colors="#555555")
        cbar.outline.set_edgecolor("#cccccc")
        return cbar

    def _build_live_figure(self):
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 110})
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.patch.set_facecolor("white")
        fig.suptitle("Wing Spar — Live Analysis", fontsize=12,
                     fontweight="bold", color="#1a1a1a")
        fig.subplots_adjust(left=0.08, right=0.96, top=0.92,
                            bottom=0.09, hspace=0.45, wspace=0.35)

        empty = np.empty((0, 2))
        ax_disp, ax_trace, ax_twist, ax_tv = axes.flat

        # --- Displacement vs time: two fixed-colour series ---
        self._style_ax(ax_disp, "Displacement vs. Time",
                       "Time  (s)", "Displacement  (m)")
        ax_disp.axhline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        self._sc_dx = ax_disp.scatter([], [], c=self._C_DISP_H, s=7,
                                      marker="o", linewidths=0, zorder=3,
                                      label="Horizontal (X)")
        self._sc_dy = ax_disp.scatter([], [], c=self._C_DISP_V, s=7,
                                      marker="^", linewidths=0, zorder=3,
                                      label="Vertical (Y)")
        self._sc_dx.set_offsets(empty)
        self._sc_dy.set_offsets(empty)
        ax_disp.legend(fontsize=7, framealpha=0.95, edgecolor="#dddddd",
                       loc="upper left")

        # --- XY trace: plasma by time ---
        self._style_ax(ax_trace, "Tracked Path (XY)",
                       "Horizontal disp.  (m)", "Vertical disp.  (m)")
        ax_trace.set_aspect("equal", adjustable="datalim")
        ax_trace.axhline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        ax_trace.axvline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        self._sc_trace = ax_trace.scatter([], [], c=[], cmap=self._CMAP,
                                          s=8, linewidths=0, zorder=3,
                                          vmin=0, vmax=1)
        self._sc_trace.set_offsets(empty)
        self._pt_start, = ax_trace.plot([], [], "o", color="#27ae60",
                                        ms=7, zorder=6, label="Start")
        self._pt_end,   = ax_trace.plot([], [], "s", color="#e74c3c",
                                        ms=7, zorder=6, label="End")
        ax_trace.legend(fontsize=7, framealpha=0.95, edgecolor="#dddddd",
                        loc="upper left")
        self._cbar_trace = self._make_cbar(fig, self._sc_trace, ax_trace)

        # --- Twist vs time: fixed colour ---
        self._style_ax(ax_twist, "Spar Twist vs. Time",
                       "Time  (s)", "Twist  (°)")
        ax_twist.axhline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        self._sc_twist = ax_twist.scatter([], [], c=self._C_ANGLE, s=7,
                                          linewidths=0, zorder=3)
        self._sc_twist.set_offsets(empty)

        # --- Twist vs vertical displacement: plasma by time ---
        self._style_ax(ax_tv, "Twist vs. Vertical Displacement",
                       "Twist  (°)", "Vertical displacement  (m)")
        ax_tv.axhline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        ax_tv.axvline(0, color=self._C_ZERO, linewidth=0.8, zorder=1)
        self._sc_tv = ax_tv.scatter([], [], c=[], cmap=self._CMAP,
                                    s=8, linewidths=0, zorder=3,
                                    vmin=0, vmax=1)
        self._sc_tv.set_offsets(empty)
        self._cbar_tv = self._make_cbar(fig, self._sc_tv, ax_tv)

        self._live_fig = fig

    # ------------------------------------------------------------------
    def _redraw(self, dx, dy, angle, t):
        ta = np.asarray(t, dtype=float)
        tmax = max(ta[-1], 0.001)
        xy_disp_h = np.column_stack([t, dx])
        xy_disp_v = np.column_stack([t, dy])
        xy_trace  = np.column_stack([dx, dy])
        xy_twist  = np.column_stack([t, angle])
        xy_tv     = np.column_stack([angle, dy])

        # Displacement scatter
        self._sc_dx.set_offsets(xy_disp_h)
        self._sc_dy.set_offsets(xy_disp_v)

        # XY trace (cmap)
        self._sc_trace.set_offsets(xy_trace)
        self._sc_trace.set_array(ta)
        self._sc_trace.set_clim(0, tmax)
        self._pt_start.set_data([dx[0]],  [dy[0]])
        self._pt_end.set_data(  [dx[-1]], [dy[-1]])

        self._sc_twist.set_offsets(xy_twist)

        # Twist vs vertical (cmap)
        self._sc_tv.set_offsets(xy_tv)
        self._sc_tv.set_array(ta)
        self._sc_tv.set_clim(0, tmax)

        # Rescale all axes
        ax_disp = self._sc_dx.axes
        ax_disp.ignore_existing_data_limits = True
        ax_disp.update_datalim(self._sc_dx.get_datalim(ax_disp.transData))
        ax_disp.update_datalim(self._sc_dy.get_datalim(ax_disp.transData))
        ax_disp.autoscale_view()

        for sc in (self._sc_trace, self._sc_twist, self._sc_tv):
            ax = sc.axes
            ax.ignore_existing_data_limits = True
            ax.update_datalim(sc.get_datalim(ax.transData))
            ax.autoscale_view()

        self._live_fig.canvas.draw_idle()
        self._live_fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def _run(self):
        self._build_live_figure()
        self._live_fig.show()
        while not self._stop.is_set():
            self._dirty.wait(timeout=self.REDRAW_INTERVAL)
            self._dirty.clear()
            with self._lock:
                dx    = self._dx[:]
                dy    = self._dy[:]
                angle = self._angle[:]
            if dx:
                t = [i / self._fps for i in range(len(dx))]
                self._redraw(dx, dy, angle, t)
        # final redraw before saving
        with self._lock:
            dx    = self._dx[:]
            dy    = self._dy[:]
            angle = self._angle[:]
        if dx:
            t = [i / self._fps for i in range(len(dx))]
            self._redraw(dx, dy, angle, t)

    # ------------------------------------------------------------------
    def push(self, dx_mm, dy_mm, angle_deg):
        with self._lock:
            self._dx.append(dx_mm)
            self._dy.append(dy_mm)
            self._angle.append(angle_deg)
        self._dirty.set()

    def clear(self):
        with self._lock:
            self._dx.clear()
            self._dy.clear()
            self._angle.clear()
        self._dirty.set()

    # ------------------------------------------------------------------
    def _save_ax_as_image(self, suffix, stem, base, dx, dy, angle, t):
        from matplotlib.lines import Line2D
        ta = np.asarray(t, dtype=float)
        tmax = max(ta[-1], 0.001)

        right_margin = 0.88
        fig_out, ax_out = plt.subplots(figsize=(8, 5))
        fig_out.patch.set_facecolor("white")
        fig_out.subplots_adjust(left=0.12, right=right_margin,
                                top=0.88, bottom=0.13)

        meta = {
            "displacement": (
                "Wing Spar — Displacement vs. Time",
                "Time  (s)", "Displacement  (m)",
            ),
            "xy_path": (
                "Wing Spar — Tracked Path (XY Plane)",
                "Horizontal displacement  (m)", "Vertical displacement  (m)",
            ),
            "twist": (
                "Wing Spar — Spar Twist vs. Time",
                "Time  (s)", "Twist angle  (°)",
            ),
            "twist_vs_vert": (
                "Wing Spar — Twist vs. Vertical Displacement",
                "Twist angle  (°)", "Vertical displacement  (m)",
            ),
        }
        title, xlabel, ylabel = meta[suffix]
        self._style_ax(ax_out, "", xlabel, ylabel, fontsize_title=12)
        ax_out.axhline(0, color=self._C_ZERO, linewidth=0.9, zorder=1)
        fig_out.suptitle(title, fontsize=13, fontweight="bold",
                         color="#1a1a1a", y=0.97)

        def _add_cbar(sc):
            cbar = fig_out.colorbar(sc, ax=ax_out, pad=0.02, fraction=0.03)
            cbar.set_label("Time  (s)", fontsize=8, color="#444444")
            cbar.ax.tick_params(labelsize=7, colors="#555555")
            cbar.outline.set_edgecolor("#cccccc")

        if suffix == "displacement":
            fig_out.subplots_adjust(right=0.95)
            ax_out.scatter(t, dx, c=self._C_DISP_H, s=10, marker="o",
                           linewidths=0, zorder=3, label="Horizontal (X)")
            ax_out.scatter(t, dy, c=self._C_DISP_V, s=10, marker="^",
                           linewidths=0, zorder=3, label="Vertical (Y)")
            ax_out.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=self._C_DISP_H, ms=8,
                           label="Horizontal (X)"),
                    Line2D([0], [0], marker="^", color="w",
                           markerfacecolor=self._C_DISP_V, ms=8,
                           label="Vertical (Y)"),
                ],
                fontsize=8, framealpha=0.95, edgecolor="#dddddd",
            )

        elif suffix == "xy_path":
            ax_out.set_aspect("equal", adjustable="datalim")
            ax_out.axvline(0, color=self._C_ZERO, linewidth=0.9, zorder=1)
            sc = ax_out.scatter(dx, dy, c=ta, cmap=self._CMAP, s=10,
                                linewidths=0, zorder=3, vmin=0, vmax=tmax)
            ax_out.plot(dx[0],  dy[0],  "o", color="#27ae60", ms=8,
                        zorder=6, label="Start")
            ax_out.plot(dx[-1], dy[-1], "s", color="#e74c3c", ms=8,
                        zorder=6, label="End")
            ax_out.legend(fontsize=8, framealpha=0.95, edgecolor="#dddddd")
            _add_cbar(sc)

        elif suffix == "twist":
            fig_out.subplots_adjust(right=0.95)
            ax_out.scatter(t, angle, c=self._C_ANGLE, s=10,
                           linewidths=0, zorder=3)

        elif suffix == "twist_vs_vert":
            ax_out.axhline(0, color=self._C_ZERO, linewidth=0.9, zorder=1)
            ax_out.axvline(0, color=self._C_ZERO, linewidth=0.9, zorder=1)
            sc = ax_out.scatter(angle, dy, c=ta, cmap=self._CMAP, s=10,
                                linewidths=0, zorder=3, vmin=0, vmax=tmax)
            _add_cbar(sc)

        out = base / f"{stem}_{suffix}.png"
        fig_out.savefig(str(out), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig_out)
        print(f"Saved: {out}")

    def _save_cv_txt(self, stem, base, dy, t):
        out = base / f"{stem}_cv_displacement.txt"
        with out.open("w", encoding="utf-8") as f:
            f.write("time_s\tdy_m\n")
            for ti, yi in zip(t, dy):
                f.write(f"{ti:.6f}\t{yi:.6f}\n")
        print(f"Saved: {out}")

    def _save_twist_txt(self, stem, base, angle, t):
        out = base / f"{stem}_cv_twist.txt"
        with out.open("w", encoding="utf-8") as f:
            f.write("time_s\tangle_deg\n")
            for ti, ai in zip(t, angle):
                f.write(f"{ti:.6f}\t{ai:.6f}\n")
        print(f"Saved: {out}")

    def save(self, video_path):
        with self._lock:
            dx    = self._dx[:]
            dy    = self._dy[:]
            angle = self._angle[:]
        if not dx:
            return
        t = [i / self._fps for i in range(len(dx))]
        stem = Path(video_path).stem
        self._save_cv_txt(stem, DATA_DIR, dy, t)
        self._save_twist_txt(stem, DATA_DIR, angle, t)
        for suffix in ("displacement", "xy_path", "twist", "twist_vs_vert"):
            self._save_ax_as_image(suffix=suffix, stem=stem, base=FIGURES_DIR,
                                   dx=dx, dy=dy, angle=angle, t=t)

    def stop_and_save(self, video_path):
        self._stop.set()
        self._thread.join(timeout=3.0)
        with self._lock:
            dx    = self._dx[:]
            dy    = self._dy[:]
            angle = self._angle[:]
        if self._live_fig is not None:
            plt.close(self._live_fig)
        if not dx:
            return
        t = [i / self._fps for i in range(len(dx))]
        stem = Path(video_path).stem
        self._save_cv_txt(stem, DATA_DIR, dy, t)
        self._save_twist_txt(stem, DATA_DIR, angle, t)
        for suffix in ("displacement", "xy_path", "twist", "twist_vs_vert"):
            self._save_ax_as_image(suffix=suffix, stem=stem, base=FIGURES_DIR,
                                   dx=dx, dy=dy, angle=angle, t=t)


def main():
    args = parse_args()
    global MIN_TRACK_POINTS
    MIN_TRACK_POINTS = args.min_track_points
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero.")
    if args.min_track_points < 3:
        raise ValueError("--min-track-points must be at least 3.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame.")

    max_display_height = args.max_height or int(screen_height() * 0.9)
    first_frame, display_scale = fit_to_height(first_frame, max_display_height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    saved_setup = (
        load_setup(args.setup_file, first_frame.shape, args.known_height_mm)
        if args.use_saved_setup
        else None
    )

    if saved_setup is not None:
        mm_per_px = float(saved_setup["mm_per_px"])
        roi_polygon = np.array(saved_setup["roi_polygon"], dtype=np.float32)
        color_samples = (
            np.array(saved_setup["color_samples"], dtype=np.int32)
            if saved_setup.get("color_samples") is not None
            else None
        )
    else:
        mm_per_px = choose_calibration_points(first_frame, args.known_height_mm)
        roi_polygon = choose_tracking_polygon(first_frame)
        color_samples = None

    roi_mask = polygon_mask(first_gray.shape, roi_polygon)
    color_hsv = None
    if args.no_color:
        color_samples = None
    elif color_samples is None:
        color_samples = choose_color_samples(
            first_frame,
            args.hue_tolerance,
            args.sat_tolerance,
            args.val_tolerance,
        )

    if color_samples is not None:
        _, color_hsv = color_mask_from_samples(
            first_frame,
            color_samples,
            args.hue_tolerance,
            args.sat_tolerance,
            args.val_tolerance,
        )

    if saved_setup is None:
        save_setup(
            args.setup_file,
            first_frame.shape,
            args.known_height_mm,
            mm_per_px,
            roi_polygon,
            color_samples,
        )
    if args.tracker in ("color", "template") and color_hsv is None:
        raise SystemExit(
            f"--tracker {args.tracker} needs color samples. Run without --no-color."
        )
    tracker = HybridTracker(
        first_frame,
        first_gray,
        roi_mask,
        roi_polygon,
        args.max_features,
        args.search_margin,
        args.roi_scale,
        args.template_angle_range,
        args.template_angle_step,
        args.template_min_score,
        args.template_match_scale,
        color_hsv,
        (args.hue_tolerance, args.sat_tolerance, args.val_tolerance),
    )

    roi_corners = roi_polygon.reshape(-1, 1, 2)
    moments = cv2.moments(roi_polygon)
    if moments["m00"]:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    else:
        center_x, center_y = np.mean(roi_polygon, axis=0)
    start_center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)

    paused = False
    window = "Wing spar analysis"
    if args.hide_debug:
        set_video_window(window, first_frame)
    else:
        set_video_window(window, np.hstack([first_frame, first_frame]))

    status = "Space: pause | Q/Esc: quit | Speed set with --speed"
    paused_at = None
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    last_tracking = identity, 0.0, len(tracker.start_points), "initial"
    trace_points = []
    debug_frame_counter = 0
    cached_debug_view = None

    plot_worker = PlotWorker(fps)

    while True:
        if paused:
            frame = last_frame.copy()
            skipped = 0
        else:
            ok, frame, skipped = read_synced_frame(cap)
            if not ok:
                if not args.loop:
                    break
                plot_worker.save(args.video)
                cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
                tracker.reset()
                trace_points.clear()
                plot_worker.clear()
                cached_debug_view = None
                last_tracking = identity, 0.0, len(tracker.start_points), "initial"
                ok, frame = cap.read()
                skipped = 0
                if not ok:
                    break
            if display_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (first_frame.shape[1], first_frame.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            last_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if paused:
            transform, angle_deg, feature_count, method = last_tracking
        else:
            transform, angle_deg, feature_count, method = tracker.update(
                frame, gray, args.tracker
            )
            last_tracking = transform, angle_deg, feature_count, method

        tracked_corners = None
        tracked_center = None
        tracked_points = None if method == "template" else tracker.display_points
        dx_mm = 0.0
        dy_mm = 0.0
        if angle_deg is None:
            angle_deg = 0.0

        tracking_ok = transform is not None and (
            feature_count >= MIN_TRACK_POINTS
            or method.startswith("color")
            or method == "template"
        )
        if not tracking_ok:
            status = ["Tracking weak/lost. Restart and select a more textured ROI."]
        else:
            tracked_corners = cv2.transform(roi_corners, transform).reshape(-1, 2)
            tracked_center = cv2.transform(start_center, transform).reshape(2)
            if args.show_trace:
                trace_points.append(tuple(tracked_center))
                if args.trace_length > 0 and len(trace_points) > args.trace_length:
                    del trace_points[: len(trace_points) - args.trace_length]
            start_center_xy = start_center.reshape(2)
            dx_px = tracked_center[0] - start_center_xy[0]
            dy_px = tracked_center[1] - start_center_xy[1]
            dx_mm = dx_px * mm_per_px / 1000.0
            dy_mm = -dy_px * mm_per_px / 1000.0
            if not paused:
                plot_worker.push(dx_mm, dy_mm, angle_deg)
            if method == "template":
                metric_line = f"Template IoU score: {feature_count / 1000.0:.3f}"
            else:
                count_label = "area" if method.startswith("color") else "features"
                metric_line = f"Tracker: {method} ({feature_count} {count_label})"
            status = [metric_line]
            if skipped:
                status.append(f"Skipped frame(s): {skipped}")

        draw_tracking_overlay(
            frame,
            tracked_corners,
            tracked_center,
            tracked_points,
            trace_points if args.show_trace else [],
            dx_mm,
            dy_mm,
            angle_deg,
            status,
        )

        if args.hide_debug:
            display = frame
        else:
            debug_frame_counter += 1
            if debug_frame_counter % 4 == 0 or cached_debug_view is None:
                cached_debug_view = draw_live_debug_view(last_frame, tracker)
            display = np.hstack([frame, cached_debug_view])
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            if paused:
                paused_at = time.perf_counter()
            elif paused_at is not None:
                paused_at = None

    cap.release()
    cv2.destroyAllWindows()
    plot_worker.stop_and_save(args.video)


if __name__ == "__main__":
    main()

"""Multi-rotation bounding-box estimation.

The bbox head only ever regresses a single x-coordinate (xmin) from a frame.
Rather than trusting one straight-on reading and padding it out with
dataset-average height/width (the old approach), we rotate the frame through
several angles and query the model once per angle. Each reading gives us an
x-position measured along a different direction, turning into a measurement
of the object from that angle. Projected back into the original frame and
combined, these readings trace out a polygon that hugs the object.

Because the model only measures position along x, height (y) is the
uninteresting axis here: we deliberately keep the crop height fixed and let
the crop *width* grow with the rotation angle (up to the frame's diagonal),
so that as the object's silhouette widens under rotation we don't clip the
very information the model is trying to measure.

For now, testing will be done with num_rotations=4 (0, 90, 180, 270) to keep
inference time down.
"""

import cv2
import numpy as np
from scipy.spatial import ConvexHull


def pad_to_square(image, border_value=0.0):
    """Embed `image` in a square canvas sized to its diagonal, so that ANY
    rotation about the center keeps the full original content in-frame."""
    h, w = image.shape[:2]
    s = int(np.ceil(np.hypot(w, h)))
    canvas = np.full((s, s), border_value, dtype=image.dtype)
    x0, y0 = (s - w) // 2, (s - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image
    return canvas, (x0, y0)


def _rotate_canvas(padded, angle, border_value=0.0):
    s = padded.shape[0]
    center = (s / 2, s / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (s, s), borderValue=border_value)
    return rotated, M


def _dynamic_crop(rotated, angle, target_w, target_h):
    """Crop the rotated square canvas back down to the model's expected input
    shape. Width is widened per-angle to keep the whole rotated silhouette
    in view (x matters); height is kept at the model's fixed target_h
    regardless of angle (y doesn't)."""
    s = rotated.shape[0]
    rad = np.deg2rad(angle)
    needed_w = abs(target_w * np.cos(rad)) + abs(target_h * np.sin(rad))
    crop_w = int(np.clip(np.ceil(needed_w), target_w, s))
    crop_h = min(target_h, s)

    cx, cy = s // 2, s // 2
    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(s, x0 + crop_w)
    y1 = min(s, y0 + crop_h)

    crop = rotated[y0:y1, x0:x1]
    crop_resized = cv2.resize(crop, (target_w, target_h))
    # scale factors to map a coordinate in crop_resized back to rotated-canvas coords
    sx = (x1 - x0) / target_w
    sy = (y1 - y0) / target_h
    return crop_resized, (x0, y0, sx, sy)


def estimate_polygon(bbox_predict_fn, padded, pad_offset, target_w, target_h,
                      avg_center_y, avg_height, num_rotations=8):
    """Rotate through `num_rotations` angles, call `bbox_predict_fn(crop) -> xmin`
    at each, and fuse the resulting edge points into a polygon.

    `bbox_predict_fn` takes a (target_h, target_w) preprocessed crop and returns
    a predicted xmin in that crop's own pixel coordinates.

    Returns an (N, 2) array of hull vertices in the ORIGINAL (unpadded) image's
    coordinate system, or None if there weren't enough points for a hull.
    """
    pad_x, pad_y = pad_offset
    s = padded.shape[0]
    center = (s / 2, s / 2)
    cy_pad = pad_y + avg_center_y  # object's typical vertical center, in padded-canvas coords

    all_points = []
    for angle in np.linspace(0, 360, num_rotations, endpoint=False):
        rotated, _ = _rotate_canvas(padded, angle)
        crop, (x0, y0, sx, sy) = _dynamic_crop(rotated, angle, target_w, target_h)

        xmin_pred = bbox_predict_fn(crop)
        x_in_rot = x0 + xmin_pred * sx  # back into rotated-canvas coords

        # Two points spanning the object's (assumed) height at this x, in rotated-canvas coords.
        p1_rot = (x_in_rot, cy_pad - avg_height / 2)
        p2_rot = (x_in_rot, cy_pad + avg_height / 2)

        # Undo the rotation to land back in (padded, unrotated) canvas coords.
        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
        p1 = M_inv @ np.array([p1_rot[0], p1_rot[1], 1.0])
        p2 = M_inv @ np.array([p2_rot[0], p2_rot[1], 1.0])

        all_points.extend(
            ((p1[0] - pad_x, p1[1] - pad_y), (p2[0] - pad_x, p2[1] - pad_y))
        )
    pts = np.array(all_points)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    return pts[hull.vertices]
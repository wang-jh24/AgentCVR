# Frame resize, bbox scaling, optional base64 encode.

import json
import os
import cv2
import numpy as np
import base64
from typing import Union, Tuple, List, Dict

def process_frame(
        frame_path: str,
        length: int,
        encode: bool = False,
        visualize: bool = False,
        bboxes: Dict[int, Dict[str, int]] = None
) -> Tuple[Union[str, np.ndarray], Dict[int, Dict[str, int]]]:
    """Load frame, scale to max dimension length, scale bboxes; optionally encode to base64."""
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {frame_path}")
    h, w = img.shape[:2]
    max_dim = max(h, w)
    scale = length / max_dim if max_dim > length else 1.0
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
    scaled_bboxes = {}
    if bboxes is not None:
        for idx, bbox in bboxes.items():
            if bbox is None:
                scaled_bboxes[idx] = None
            else:
                scaled = {
                    'xtl': int(bbox['xtl'] * scale),
                    'ytl': int(bbox['ytl'] * scale),
                    'xbr': int(bbox['xbr'] * scale),
                    'ybr': int(bbox['ybr'] * scale)
                }
                scaled_bboxes[idx] = scaled
    if encode:
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image, scaled_bboxes
    else:
        return img, scaled_bboxes


def process_frames(frame_dir, n_frames, bboxes, length=640, encode=True, visualize=False):
    """Sample n_frames from frame_dir, process each with bbox; returns (frames, bboxes, mapping_dict)."""
    frames = sorted(os.listdir(frame_dir))
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    mapping_dict = {i: int(indices[i]) for i in range(len(indices))}
    frame_paths = [os.path.join(frame_dir, frames[i]) for i in indices]
    processed_frames = []
    processed_bboxes = []
    for frame_path, original_frame_index in zip(frame_paths, indices):
        input_bboxes = {}
        for idx, bbox in bboxes.items():
            if bbox is None:
                input_bboxes[idx] = None
            else:
                input_bboxes[idx] = bbox.get(str(original_frame_index))
        processed_frame, processed_bbox = process_frame(frame_path, length, encode, visualize, input_bboxes)
        processed_frames.append(processed_frame)
        processed_bboxes.append(processed_bbox)
    return processed_frames, processed_bboxes, mapping_dict

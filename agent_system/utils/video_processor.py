# Video sampling by time intervals; supports URL download and multi-segment intervals.

import os
import shutil
import tempfile
import re
import bisect
import math
import cv2
import numpy as np
import requests
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download
import base64

def _allocate_frames(durations, total_frames):
    """Allocate total_frames across segments by duration proportion."""
    total_duration = sum(durations)
    if total_duration == 0:
        return [0] * len(durations)
    ideal_frames = [(d / total_duration) * total_frames for d in durations]
    integer_parts = [math.floor(x) for x in ideal_frames]
    fractions = [x - math.floor(x) for x in ideal_frames]
    total_integer = sum(integer_parts)
    remaining = total_frames - total_integer
    if remaining < 0:
        return integer_parts
    sorted_indices = sorted(range(len(fractions)), key=lambda i: -fractions[i])
    for i in range(remaining):
        idx = sorted_indices[i]
        integer_parts[idx] += 1
    return integer_parts

def process_video(input_path: str, n_frames: int, intervals: list, max_length: int = None, encode: bool = True) -> (list, list, list, float, int, float):
    """Process one video: n_frames>0 sample uniformly in intervals; n_frames=-1 read all frames. Returns (frame_groups, mapping_tables, new_fps_list, original_fps, total_frames, duration)."""
    is_temp, temp_dir, video_path = False, None, None
    if input_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(input_path, stream=True)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            video_path = temp_file.name
            is_temp = True
        except requests.exceptions.RequestException as e:
            print(f"  Error: video download failed {input_path}: {e}")
            return [], [], [], 0.0, 0, 0.0
    else:
        video_path = input_path

    if not video_path or not os.path.exists(video_path):
        if is_temp and video_path:
            os.unlink(video_path)
        print(f"  Error: video file not found: {input_path}")
        return [], [], [], 0.0, 0, 0.0

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames_video = len(vr)
        original_fps = vr.get_avg_fps() if vr.get_avg_fps() > 0 else 30.0
        video_duration = total_frames_video / original_fps if original_fps > 0 else 0
        if intervals is None:
            intervals = [[(0, video_duration)]]
        if total_frames_video == 0:
            print(f"  Warning: video has 0 frames: {input_path}")
            return [], [], [], original_fps, 0, 0.0

        parsed_intervals = []
        for interval in intervals:
            segs = interval if isinstance(interval[0], (list, tuple)) else [interval]
            valid_segs = []
            try:
                for seg in segs:
                    if len(seg) != 2:
                        continue
                    start_t = float(seg[0])
                    end_t = float(seg[1]) if seg[1] is not None else video_duration
                    if start_t < end_t:
                        valid_segs.append((start_t, end_t))
            except (ValueError, TypeError):
                continue
            if valid_segs:
                parsed_intervals.append(valid_segs)

        if not parsed_intervals:
            return [], [], [], original_fps, total_frames_video, video_duration

        all_processed_frames = []
        all_mapping_tables = []
        all_new_fps = []
        read_all_frames_mode = (n_frames == -1)
        if not read_all_frames_mode:
            durations = [sum(end - start for start, end in segs) for segs in parsed_intervals]
            if sum(durations) == 0:
                return [], [], [], original_fps, total_frames_video, video_duration
            allocated_frames = _allocate_frames(durations, n_frames)
        else:
            allocated_frames = [0] * len(parsed_intervals)

        for i, segs in enumerate(parsed_intervals):
            frame_indices = []
            if read_all_frames_mode:
                for start_t, end_t in segs:
                    start_idx = int(start_t * original_fps)
                    end_idx = int(end_t * original_fps)
                    frame_indices.extend(range(start_idx, min(end_idx, total_frames_video)))
                frame_indices = sorted(list(set(frame_indices)))
            else:
                k = allocated_frames[i]
                if k <= 0:
                    all_processed_frames.append([])
                    all_mapping_tables.append({})
                    all_new_fps.append(0.0)
                    continue
                seg_durations = [end - start for start, end in segs]
                cum_dur = np.cumsum([0.0] + seg_durations)
                total_seg_duration = cum_dur[-1]
                virtual_times = np.linspace(0, total_seg_duration, num=k, endpoint=False) + (total_seg_duration / (2 * k))
                actual_times = []
                for t in virtual_times:
                    idx = bisect.bisect_right(cum_dur, t) - 1
                    idx = max(0, min(idx, len(segs) - 1))
                    seg_start, _ = segs[idx]
                    t_in_seg = t - cum_dur[idx]
                    actual_time = min(seg_start + t_in_seg, video_duration)
                    actual_times.append(actual_time)
                frame_indices = sorted(list(set([min(int(t * original_fps), total_frames_video - 1) for t in actual_times])))

            if not frame_indices:
                all_processed_frames.append([])
                all_mapping_tables.append({})
                all_new_fps.append(0.0)
                continue
            try:
                frames = vr.get_batch(frame_indices).asnumpy()
            except Exception as e:
                print(f"  Error: decord get_batch failed: {e}")
                all_processed_frames.append([])
                all_mapping_tables.append({})
                all_new_fps.append(0.0)
                continue
            processed_frames = []
            for frame in frames:
                if max_length and max(frame.shape[:2]) > max_length:
                    scale = max_length / max(frame.shape[:2])
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                if encode:
                    success, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if success:
                        processed_frames.append(base64.b64encode(buffer).decode('utf-8'))
                else:
                    processed_frames.append(frame)
            mapping_table = {i: frame_idx for i, frame_idx in enumerate(frame_indices)}
            interval_duration = sum(end - start for start, end in segs)
            new_fps = len(processed_frames) / interval_duration if interval_duration > 0 else 0
            all_processed_frames.append(processed_frames)
            all_mapping_tables.append(mapping_table)
            all_new_fps.append(new_fps)
        return all_processed_frames, all_mapping_tables, all_new_fps, original_fps, total_frames_video, video_duration
    finally:
        if is_temp and video_path and os.path.exists(video_path):
            if temp_dir:
                shutil.rmtree(temp_dir)
            else:
                os.unlink(video_path)


def process_videos_for_question(pair: dict, n_frames_per_video: int, length: int):
    """Process all videos for one QA pair; returns (all_frames_base64, frames_str_placeholder)."""
    from .config import get_local_video_root, get_remote_video_base_url
    all_frames_base64 = []
    frames_str_placeholder = ""
    for idx, v_name in enumerate(pair["videos"]):
        begin, end = int(pair["begin"][idx]), int(pair["end"][idx])
        local_path = os.path.join(get_local_video_root(), "PEA", f"{v_name.replace('/', '_')}.mp4")
        remote_base = get_remote_video_base_url().rstrip("/")
        video_input_path = local_path if os.path.exists(local_path) else f"{remote_base}/PEA/{v_name}.mp4"
        try:
            frame_groups, mapping_tables, new_fps_list, original_fps, total_frames, duration = process_video(
                input_path=video_input_path,
                n_frames=n_frames_per_video,
                intervals=[[[begin, end]]],
                max_length=length,
                encode=True
            )
            if not frame_groups or not frame_groups[0]:
                continue
            video_frames = frame_groups[0]
            all_frames_base64.extend(video_frames)
            frame_placeholder = "".join("<frame>" for _ in range(len(video_frames)))
            frames_str_placeholder += f"Video {idx + 1}:\n{frame_placeholder}\n"
        except Exception as e:
            print(f"   Error processing video {idx+1}: {e}")
            return None, None
    return all_frames_base64, frames_str_placeholder

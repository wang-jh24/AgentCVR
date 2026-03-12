# Path and API config from environment variables.

import os


def get_local_video_root() -> str:
    """Local video cache root; subdirs include CC, NC, BU, PEA, etc. (task codes)."""
    return os.environ.get("LOCAL_VIDEO_ROOT", "/path/to/your/local/video/cache")


def get_remote_video_base_url() -> str:
    """Remote video base URL; scripts append paths like /youcook2/xxx.mp4."""
    return os.environ.get("REMOTE_VIDEO_BASE_URL", "https://your-remote-video-server.com")


def get_uav_data_dir() -> str:
    """UAV dataset root directory."""
    return os.environ.get("UAV_DATA_DIR", "/path/to/your/uav/dataset")


def build_video_paths(subdir: str, video_name: str, suffix: str = ".mp4") -> tuple:
    """Build (local_path, remote_url) for a video name under subdir."""
    local_root = get_local_video_root()
    remote_base = get_remote_video_base_url().rstrip("/")
    local_path = os.path.join(local_root, subdir, f"{video_name}{suffix}")
    remote_url = f"{remote_base}/{subdir}/{video_name}{suffix}"
    return local_path, remote_url

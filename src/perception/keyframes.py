import os
import ffmpeg

def extract_keyframes(video_path: str, out_dir: str, fps: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .output(os.path.join(out_dir, 'frame_%06d.jpg'), vsync='vfr')
        .overwrite_output()
        .run(quiet=True)
    )
    return sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")])

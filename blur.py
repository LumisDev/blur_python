import cv2
import numpy as np
from numba import njit, prange
from moviepy import VideoFileClip, ImageSequenceClip
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="Input file")
parser.add_argument("ofile", type=str, help="Output file")
args = parser.parse_args()

# Step 2: Optimized Motion Blur with Color Conservation
@njit(parallel=True, fastmath=True, cache=True)
def apply_temporal_motion_blur(prev_frames, curr_frame, next_frames):
    """Blends five frames (2 previous, current, and 2 next) while preserving original colors using perceptual weighting."""
    output_frame = np.empty_like(curr_frame, dtype=np.uint8)

    for i in prange(curr_frame.shape[0]):  # Height
        for j in prange(curr_frame.shape[1]):  # Width
            for c in prange(curr_frame.shape[2]):  # Color channels
                # Weighted blending of the 5 frames
                blended_pixel = (
                    0.2 * prev_frames[0, i, j, c] +
                    0.2 * prev_frames[1, i, j, c] +
                    0.4 * curr_frame[i, j, c] +
                    0.2 * next_frames[0, i, j, c] +
                    0.2 * next_frames[1, i, j, c]
                )
                output_frame[i, j, c] = min(255, max(0, int(blended_pixel)))

    return output_frame

# Step 3: Read the input video into memory
video_path = args.infile
video_capture = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Step 4: Load all frames into memory
frames = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frames.append(frame)
video_capture.release()  # Release video file

frames = np.array(frames, dtype=np.uint8)
frame_count = len(frames)
print(f"Loaded {frame_count} frames into memory.")

# Step 5: Apply motion blur in memory
processed_frames = []

for i in tqdm(range(frame_count), desc="Processing frames...", unit="frame"):
    # Handle edge cases by duplicating nearest available frames
    prev_frames = [
        frames[max(0, i - 2)],  # Use frame 0 if i - 2 is out of bounds
        frames[max(0, i - 1)],  # Use frame 0 if i - 1 is out of bounds
    ]
    curr_frame = frames[i]
    next_frames = [
        frames[min(frame_count - 1, i + 1)],  # Use last frame if i + 1 is out of bounds
        frames[min(frame_count - 1, i + 2)],  # Use last frame if i + 2 is out of bounds
    ]

    # Apply motion blur with 2 previous and 2 next frames
    blurred_frame = apply_temporal_motion_blur(np.array(prev_frames), curr_frame, np.array(next_frames))

    processed_frames.append(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))

# Step 6: Convert processed frames to video
print("Saving processed video...")
processed_sequence = ImageSequenceClip(processed_frames, fps=fps)

# Step 7: Add original audio back
original_video = VideoFileClip(video_path)
processed_clip:VideoFileClip = processed_sequence.with_audio(original_video.audio)

# Save final output
processed_clip.write_videofile(args.ofile, codec="libx264", fps=fps, audio_codec="libmp3lame")

print(f"Processing complete! Saved as '{args.ofile}'")

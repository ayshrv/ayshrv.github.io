import os
import cv2
import glob
import numpy as np
# import ft2
from PIL import ImageFont, ImageDraw, Image
from pdf2image import convert_from_path

fontpath = "/Users/ayshrv/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Avenir-Medium.ttf"
font = ImageFont.truetype(fontpath, 36)

def combine_videos_grid(thermal_paths, kaist_paths, output_path, frame_size=(1064//2, 1114//2)):
    """
    Combine 4 thermal_im and 4 kaist videos into a single video in a 2x4 grid with 20px padding between frames.
    Top row: [T1, T2, K1, K2] (timestep 1), Bottom row: [T3, T4, K3, K4] (timestep 2)
    For each frame t, if t%2==0 add 'RGB' below every frame, else add 'Thermal' below every frame.
    """
    assert len(thermal_paths) == 4 and len(kaist_paths) == 4, "Need 4 videos from each source."
    
    # Open all video captures
    caps = [cv2.VideoCapture(p) for p in thermal_paths + kaist_paths]
    
    # Get min frame count
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    min_frames = min(frame_counts)
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    
    text_band_height = 50
    padding = 20
    frame_w, frame_h = frame_size[0], frame_size[1] + text_band_height
    # 4 frames + 3 paddings per row, 2 rows + 1 padding between rows
    grid_width = frame_w * 4 + padding * 3
    grid_height = frame_h * 2 + padding
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    for t in range(min_frames):
        frames = []
        label = "RGB" if t % 2 == 0 else "Thermal"
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            else:
                # trim by 50 px from bottom
                frame = frame[:-50, :]
                frame = cv2.resize(frame, frame_size)
            # Add text band below
            canvas = np.ones((frame_size[1] + text_band_height, frame_size[0], 3), dtype=np.uint8) * 255
            canvas[:frame_size[1], :] = frame
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (frame_size[0] - text_width) // 2
            text_y = frame_size[1] + (text_band_height - text_height) // 2
            draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))
            frames.append(np.array(img_pil))
        # Top row: [T1, T2, K1, K2], Bottom row: [T3, T4, K3, K4]
        def row_with_padding(indices):
            row_frames = [frames[i] for i in indices]
            padded_row = row_frames[0]
            for f in row_frames[1:]:
                vpad = np.ones((frame_h, padding, 3), dtype=np.uint8) * 255
                padded_row = np.hstack([padded_row, vpad, f])
            return padded_row
        top_row = row_with_padding([0, 1, 4, 5])
        bottom_row = row_with_padding([2, 3, 6, 7])
        hpad = np.ones((padding, grid_width, 3), dtype=np.uint8) * 255
        grid = np.vstack([top_row, hpad, bottom_row])
        out.write(grid)
    
    # Release everything
    for cap in caps:
        cap.release()
    out.release()
    print(f"Saved grid video to {output_path}")



diff_texts = [
    "RGB", "Thermal"
]

thermal_im_data_root = "thermal_im"
kaist_data_root = "kaist"

# read all videos in the folder
kaist_video_paths = glob.glob(os.path.join(kaist_data_root, '*.mp4'))
thermal_im_video_paths = glob.glob(os.path.join(thermal_im_data_root, '*.mp4'))

# sort
kaist_video_paths = sorted(kaist_video_paths)
thermal_im_video_paths = sorted(thermal_im_video_paths)

# now change the order to be [,4,8,12], [1,5,9,13]
kaist_video_paths = kaist_video_paths[::3] + kaist_video_paths[1::3] \
    + kaist_video_paths[2::3]
thermal_im_video_paths = thermal_im_video_paths[::3] + thermal_im_video_paths[1::3] \
    + thermal_im_video_paths[2::3]


batch_size = 4
num_batches = min(len(thermal_im_video_paths), len(kaist_video_paths)) // batch_size

for i in range(num_batches):
    thermal_batch = thermal_im_video_paths[i*batch_size:(i+1)*batch_size]
    kaist_batch = kaist_video_paths[i*batch_size:(i+1)*batch_size]
    output_path = f"rgb_thermal_grid_batch_{i+1}.mp4"
    combine_videos_grid(thermal_batch, kaist_batch, output_path)





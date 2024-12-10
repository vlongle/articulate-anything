from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import base64
from rich.syntax import Syntax
from rich.jupyter import print
import numpy as np
import cv2
from PIL import Image, ImageChops
from moviepy import VideoFileClip
import os
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from PIL import Image
import math
from articulate_anything.utils.utils import HideOutput


def draw_frame(img, color, frame_width=30):
    """Draw a colored frame around an image."""
    if color:
        img_with_frame = Image.new(
            "RGB", (img.width + 2 * frame_width,
                    img.height + 2 * frame_width), color
        )
        img_with_frame.paste(img, (frame_width, frame_width))
        return img_with_frame
    else:
        return img


def display_frames(
    frames,
    titles=None,
    cols=5,
    figsize=(20, 10),
    border_color=None,
    border_width=20,
    wspace=0.0,
    hspace=0.0,
    save_file=None,
):
    """
    Display a list of frames with optional titles and optional colored borders.

    Parameters:
    - frames (list): List of frames to display.
    - titles (list): Optional list of titles for each frame.
    - cols (int): Number of columns in the display grid.
    - figsize (tuple): Size of the figure.
    - border_color (str): Optional color for the border around each frame.
    - border_width (int): Width of the border around each frame.
    - wspace (float): Width space between subplots.
    - hspace (float): Height space between subplots.
    """
    num_frames = len(frames)
    # Calculate the number of rows needed
    rows = (num_frames + cols - 1) // cols

    if border_color:
        frames = [draw_frame(frame, border_color, border_width)
                  for frame in frames]

    plt.figure(figsize=figsize)
    plt.ioff()
    for i, frame in enumerate(frames):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(frame)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()  # Close the figure after saving
    else:
        plt.show()


def show_images(images):
    return display_frames([Image.open(img) for img in images])


def convert_mp4_to_gif(input_path, output_path, start_time=0, end_time=None, resize=None, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return output_path

    with HideOutput():
        # Load the video file
        clip = VideoFileClip(input_path)
        # .subclip(start_time, end_time)

        # Resize if needed
        if resize:
            clip = clip.resize(resize)

        # Attempt a simpler write_gif call
        clip.write_gif(output_path, fps=10)
    return output_path


def show_video(video, overwrite=True, use_gif=False,
               num_frames=5, flip_horizontal=False):
    from IPython.display import Video, Image
    if use_gif:
        gif = convert_mp4_to_gif(video, video.replace(".mp4", ".gif"),
                                 overwrite=overwrite)
        display(Image(gif))
    else:
        frames = get_frames_from_video(video, to_crop_white=True,
                                       num_frames=num_frames,
                                       flip_horizontal=flip_horizontal)
        display_frames(frames, cols=5)


def show_gif(fname, width=None, height=None):
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')

    style = ""
    if width:
        style += f"width:{width}px;"
    if height:
        style += f"height:{height}px;"

    return f'<img src="data:image/gif;base64,{b64}" style="{style}" />'


def show_videos(videos, cols=3, width=None, height=None):
    if not isinstance(videos, list):
        videos = [videos]

    videos = [convert_mp4_to_gif(video, video.replace(
        ".mp4", ".gif"), overwrite=True) for video in videos]

    html_output = "<table>"
    for i, video in enumerate(videos):
        if i % cols == 0:
            html_output += "<tr>"

        html_output += "<td style='padding: 10px;'>"
        gif_path = video.replace(".mp4", ".gif")

        if os.path.exists(gif_path):
            html_output += show_gif(gif_path, width, height)
        else:
            html_output += "GIF not found"

        html_output += "</td>"

        if (i + 1) % cols == 0 or i == len(videos) - 1:
            html_output += "</tr>"

    html_output += "</table>"
    display(HTML(html_output))


def concatenate_frames_horizontally(frames):
    """
    Concatenates frames into a single image horizontally.

    Args:
        frames (list): List of PIL Images or numpy arrays to be concatenated.

    Returns:
        np.array: Concatenated image.
    """
    # Convert PIL Images to numpy arrays if necessary
    if isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]

    frames = np.array(frames)

    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(
            "The frames array must have shape (n, height, width, channels)."
        )

    concatenated_image = np.concatenate(frames, axis=1)
    return concatenated_image


def crop_white(image):
    """
    Crop white space from around a PIL image

    :param image: PIL Image object
    :return: Cropped PIL Image object
    """
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get the bounding box of the non-white area
    bg = Image.new(image.mode, image.size, (255, 255, 255))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    if bbox:
        return image.crop(bbox)
    return image  # return the original image if it's all white


def extract_frames(video_path, method="fixed", num_frames=5, interval=1):
    """
    Extract frames from a video either based on a fixed number of frames or at regular intervals.

    Parameters:
    - video_path (str): Path to the video file.
    - method (str): Method to extract frames ('fixed' or 'interval').
    - num_frames (int): Number of frames to extract (used if method is 'fixed').
    - interval (int): Interval in seconds between frames (used if method is 'interval').

    Returns:
    - frames (list): List of extracted frames.
    - frame_info (dict): Dictionary with video and frame extraction details.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    if method == "fixed":
        # Sample a fixed number of frames
        sample_indices = [int(frame_count * i / num_frames)
                          for i in range(num_frames)]
    elif method == "interval":
        # Sample frames at regular intervals
        sample_indices = [
            int(fps * i * interval) for i in range(int(duration / interval))
        ]
    else:
        raise ValueError("Invalid method. Use 'fixed' or 'interval'.")

    for idx in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()

    frame_info = {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "extracted_frame_indices": sample_indices,
    }
    frames = frames
    return frames, frame_info


def get_frames_from_video(
    video_path,
    num_frames=5,
    video_encoding_strategy="individual",
    to_crop_white=True,
    flip_horizontal=False,
    width=None,
    height=None,

):
    frames, _ = extract_frames(video_path, num_frames=num_frames)
    pil_frames = [Image.fromarray(frame) for frame in frames]

    if flip_horizontal:
        pil_frames = [frame.transpose(Image.FLIP_LEFT_RIGHT)
                      for frame in pil_frames]

    if to_crop_white:
        pil_frames = [crop_white(frame) for frame in pil_frames]

    if width is not None or height is not None:
        # Resize the frames if either width or height is specified
        pil_frames = [resize_frame(frame, width, height)
                      for frame in pil_frames]

    if video_encoding_strategy == "concatenate":
        return [Image.fromarray(concatenate_frames_horizontally(pil_frames))]
    elif video_encoding_strategy == "individual":
        return pil_frames
    else:
        raise ValueError(
            "Invalid video_encoding_strategy. Use 'concatenate' or 'individual'."
        )


def resize_frame(frame, width, height):
    if width is None and height is None:
        return frame

    original_width, original_height = frame.size

    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = original_width / original_height
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = original_height / original_width
        height = int(width * aspect_ratio)

    return frame.resize((width, height), Image.LANCZOS)


def display_code(code_string, language="python", theme="monokai", title=None):
    syntax = Syntax(code_string, language, theme=theme)
    if title:
        panel = Panel(syntax, title=title, expand=False, border_style="blue")
        Console().print(panel)
    else:
        Console().print(syntax)


def display_codes(code_snippets, languages=None, theme="monokai", titles=None, cols=2):
    if not isinstance(code_snippets, list):
        code_snippets = [code_snippets]

    if languages is None:
        languages = ["python"] * len(code_snippets)
    elif not isinstance(languages, list):
        languages = [languages] * len(code_snippets)

    if titles is None:
        titles = [None] * len(code_snippets)
    elif not isinstance(titles, list):
        titles = [titles] * len(code_snippets)

    table = Table(box=box.MINIMAL, padding=0, expand=False)
    for _ in range(cols):
        table.add_column()

    for i in range(0, len(code_snippets), cols):
        row = []
        for j in range(cols):
            if i + j < len(code_snippets):
                syntax = Syntax(
                    code_snippets[i+j], languages[i+j], theme=theme)
                if titles[i+j]:
                    panel = Panel(
                        syntax, title=titles[i+j], expand=False, border_style="blue")
                    row.append(panel)
                else:
                    row.append(syntax)
            else:
                row.append("")
        table.add_row(*row)

    Console().print(table)

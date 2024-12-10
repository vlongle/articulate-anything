from PIL import Image
import logging
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
import cv2
import os
import torch
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import numpy as np
from sklearn.cluster import KMeans
import imageio.v3 as iio
from articulate_anything.utils.utils import join_path

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE = "cool"


def get_moving_points_mask(pred_tracks, displacement_threshold=None, max_moving_points=None):
    """
    Determine which points have moved significantly on a frame-by-frame basis.

    Args:
    pred_tracks (torch.Tensor): Predicted tracks with shape (1, num_frames, num_points, 2)
    displacement_threshold (float): Minimum total displacement to consider a point as moving
    max_moving_points (int): Maximum number of moving points to return

    Returns:
    np.ndarray: Boolean mask of shape (num_points,) where True indicates a moving point
    np.ndarray: Array of shape (num_moving_points, 2) with y, x coordinates of moving points
    """
    pred_tracks_np = pred_tracks.squeeze().cpu().numpy()

    num_frames, num_points, _ = pred_tracks_np.shape

    # Calculate frame-by-frame displacement
    frame_displacements = np.linalg.norm(
        pred_tracks_np[1:] - pred_tracks_np[:-1], axis=-1
    )

    # Calculate total displacement for each track
    total_displacement = np.sum(frame_displacements, axis=0) / num_frames

    X = total_displacement.reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    # Identify the cluster with the highest mean as the "moving" cluster
    moving_cluster = kmeans.cluster_centers_.argmax()

    # Create mask for tracks in the moving cluster
    moving_mask = kmeans.labels_ == moving_cluster
    if displacement_threshold is not None:
        print("Thresholding at ", displacement_threshold)
        moving_mask = (total_displacement > displacement_threshold) & moving_mask

    # If max_moving_points is set, select top N points based on displacement
    if max_moving_points is not None and max_moving_points < np.sum(moving_mask):
        print("Getting top ", max_moving_points, " moving points")
        top_n_indices = np.argsort(total_displacement)[-max_moving_points:]
        new_mask = np.zeros_like(moving_mask)
        new_mask[top_n_indices] = True
        moving_mask = new_mask & moving_mask

    return moving_mask, total_displacement



def spatial_downsample(video, scale_factor):
    new_size = (int(video.shape[2] * scale_factor),
                int(video.shape[1] * scale_factor))
    return np.array([cv2.resize(frame, new_size) for frame in video])


def temporal_subsample(video, k):
    """
    Subsample video by selecting every kth frame.

    Args:
    video (np.array): Video array of shape (num_frames, height, width, channels)
    k (int): Sampling interval

    Returns:
    np.array: Subsampled video
    """
    return video[::k]


class DataAugmentation:
    def __init__(self, cfg):
        self.cfg = cfg
        if not isinstance(self.cfg.checkpoint_path, str) or not os.path.exists(self.cfg.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at provided path: {self.cfg.checkpoint_path}")
        self.model = CoTrackerPredictor(checkpoint=self.cfg.checkpoint_path)
        self.model = self.model.to(DEFAULT_DEVICE)

    def get_prediction(
        self, video_path, grid_size=20, down_sample=False, seg_mask_path=None,
    ):
        video = read_video_from_path(video_path)
        if down_sample:
            logging.info("COTRACKER: Downsampling...")
            video = spatial_downsample(video, scale_factor=0.1)
            video = temporal_subsample(video, k=10)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video = video.to(DEFAULT_DEVICE)
        logging.info(f">> Video moved to device: {video.device}")

        if seg_mask_path is None:
            pred_tracks, pred_visibility = self.model(
                video, grid_size=grid_size,)
        else:
            segm_mask = np.array(Image.open(seg_mask_path))[:, :, 0]
            segm_mask = torch.from_numpy(
                segm_mask)[None, None].to(DEFAULT_DEVICE)
            pred_tracks, pred_visibility = self.model(
                video, grid_size=grid_size, segm_mask=segm_mask,
            )
        return video, pred_tracks, pred_visibility

    def forward(
        self,
        video_path,
        out_video_path=None,
        grid_size=20,
        render_all=False,
        down_sample=False,
        seg_mask_path=None,
        displacement_threshold=None,
        max_moving_points=None,
        linewidth=5,
        overwrite=False,
        **kwargs,
    ):
        if out_video_path is None:
            # out_video_path is `aug_` + video_path
            video_file_name = os.path.basename(video_path)
            out_video_path = join_path(
                os.path.dirname(video_path), f"aug_{video_file_name}"
            )
        out_video_path = out_video_path.replace(
            ".mp4", "")  # remove .mp4 extension

        result_path = video_path.replace(".mp4", "_cotracker.npy")
        if os.path.exists(f"{out_video_path}.mp4") and not overwrite:
            logging.info(f"Output video already exists at {out_video_path}.mp4. Skipping processing.")
            results = np.load(result_path, allow_pickle=True).item()
            return (torch.from_numpy(results['pred_tracks']), 
                torch.from_numpy(results['pred_visibility']),
                results['moving_mask'],
                results['total_displacement'])



        video, pred_tracks, pred_visibility = self.get_prediction(
            video_path,
            grid_size=grid_size,
            down_sample=down_sample,
            seg_mask_path=seg_mask_path,
        )
        moving_mask, total_displacement = get_moving_points_mask(
            pred_tracks, displacement_threshold=displacement_threshold, 
            max_moving_points=max_moving_points)

        vis = Visualizer(
            mode=MODE,
            pad_value=120,
            linewidth=linewidth,
            save_dir=".",
            tracks_leave_trace=-1,
        )

        if render_all:
            vis.visualize(
                video.cpu(),
                pred_tracks.cpu(),  # Only visualize moving tracks
                pred_visibility.cpu(),  # Only visualize moving tracks
                filename=out_video_path,
            )
        else:
            vis.visualize(
                video.cpu(),
                # Only visualize moving tracks
                pred_tracks[:, :, moving_mask].cpu(),
                pred_visibility[
                    :, :, moving_mask
                ].cpu(),  # Only visualize moving tracks
                filename=out_video_path,
            )

        # Save results
        results = {
            "pred_tracks": pred_tracks.cpu().numpy(),
            "pred_visibility": pred_visibility.cpu().numpy(),
            "moving_mask": moving_mask,
            "total_displacement": total_displacement
        }
        np.save(result_path, results)

        return pred_tracks, pred_visibility, moving_mask, total_displacement


class OnlineDataAugmentation:
    def __init__(self, cfg):
        self.cfg = cfg
        if not isinstance(self.cfg.checkpoint_path, str) or not os.path.exists(self.cfg.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at provided path: {self.cfg.checkpoint_path}")
        self.model = CoTrackerOnlinePredictor(
            checkpoint=self.cfg.checkpoint_path)
        self.model = self.model.to(DEFAULT_DEVICE)

    def process_step(self, window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2:]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    def get_prediction(self, video_path, grid_size=20):
        window_frames = []
        is_first_step = True
        for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
            logging.debug(f"COTRACKER: Processing frame {i}...")
            if i % self.model.step == 0 and i != 0:
                pred_tracks, pred_visibility = self.process_step(
                    window_frames,
                    is_first_step,
                    grid_size=grid_size,
                    grid_query_frame=0,
                )
                is_first_step = False
            window_frames.append(frame)

        # Processing the final video frames
        pred_tracks, pred_visibility = self.process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1:],
            is_first_step,
            grid_size=grid_size,
            grid_query_frame=0,
        )
        # Combine predictions
        # pred_tracks = torch.cat(pred_tracks_list, dim=1)
        # pred_visibility = torch.cat(pred_visibility_list, dim=1)

        video = torch.tensor(np.stack(window_frames),
                             device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
        return video, pred_tracks, pred_visibility

    def forward(self, video_path, out_video_path=None, grid_size=20, render_all=False,
                seg_mask_path=None, displacement_threshold=None, max_moving_points=None,
                linewidth=5,
                overwrite=False,
                **kwargs):
        if out_video_path is None:
            # out_video_path is `aug_` + video_path
            video_file_name = os.path.basename(video_path)
            out_video_path = join_path(
                os.path.dirname(video_path), f"aug_{video_file_name}"
            )

        result_path = video_path.replace(".mp4", "_cotracker.npy")
        if os.path.exists(f"{out_video_path}.mp4") and not overwrite:
            logging.info(f"Output video already exists at {out_video_path}.mp4. Skipping processing.")
            results = np.load(result_path, allow_pickle=True).item()
            return (torch.from_numpy(results['pred_tracks']), 
                torch.from_numpy(results['pred_visibility']),
                results['moving_mask'],
                results['total_displacement'])


        segm_mask = None
        if seg_mask_path:
            segm_mask = np.array(Image.open(seg_mask_path))[:, :, 0]

        out_video_path = out_video_path.replace(
            ".mp4", "")  # remove .mp4 extension

        video, pred_tracks, pred_visibility = self.get_prediction(
            video_path,
            grid_size=grid_size,
        )
        moving_mask, total_displacement = get_moving_points_mask(pred_tracks,
                                                                 displacement_threshold=displacement_threshold,
                                                                 max_moving_points=max_moving_points)

        vis = Visualizer(
            mode=MODE,
            pad_value=120,
            linewidth=linewidth,
            save_dir=".",
            tracks_leave_trace=-1,
        )

        if render_all:
            vis.visualize(
                video.cpu(),
                pred_tracks.cpu(),
                pred_visibility.cpu(),
                filename=out_video_path,
            )
        else:
            if segm_mask is not None:
                # Check if segm_mask shape is compatible with video shape
                # segm_mask shape: (H, W); video shape: (1, T, C, H, W)
                segm_mask = torch.from_numpy(
                    segm_mask)[None, None]

                assert video.shape[3:5] == segm_mask.shape[2:
                                                           4], "Video and mask dimensions do not match"
                if torch.isnan(segm_mask).any():
                    raise ValueError(
                        "NaN values detected in segmentation mask")

                if torch.isnan(pred_tracks).any():
                    raise ValueError("NaN values detected in predicted tracks")

                vis.visualize(
                    video.cpu(),
                    pred_tracks[:, :, moving_mask].cpu(),
                    pred_visibility[:, :, moving_mask].cpu(),
                    filename=out_video_path,
                    segm_mask=segm_mask,
                    compensate_for_camera_motion=False,
                )

            else:
                vis.visualize(
                    video.cpu(),
                    pred_tracks[:, :, moving_mask].cpu(),
                    pred_visibility[:, :, moving_mask].cpu(),
                    filename=out_video_path,
                )

        # Save results
        results = {
            "pred_tracks": pred_tracks.cpu().numpy(),
            "pred_visibility": pred_visibility.cpu().numpy(),
            "moving_mask": moving_mask,
            "total_displacement": total_displacement
        }
        np.save(result_path, results)

        return pred_tracks, pred_visibility, moving_mask, total_displacement


def make_cotracker(cfg):
    CoTrackerCls = {
        "offline": DataAugmentation,
        "online": OnlineDataAugmentation,
    }
    return CoTrackerCls[cfg.cotracker.mode](cfg.cotracker)

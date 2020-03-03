#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli, filter_frame_corners


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


max_corners = 2000
block_size = 15
gf_params = dict(
    qualityLevel=0.1,
    minDistance=13,
    blockSize=block_size)


def image_to_uint8(image):
    return np.uint8(image * 255)


def create_points_mask(image, corners):
    mask = np.full_like(image, 255, dtype=np.uint8)
    for p in corners.points:
        cv2.circle(mask, (p[0], p[1]), int(gf_params["minDistance"]), 0, -1)
    return mask


def concat_corners(corners, new_corners):
    result = FrameCorners(
        np.concatenate((corners.ids, new_corners.ids)),
        np.concatenate((corners.points, new_corners.points)),
        np.concatenate((corners.sizes, new_corners.sizes)))
    return result


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    image_0 = frame_sequence[0]
    i0 = image_to_uint8(image_0)


    num_of_pix = frame_sequence.frame_shape[0] * frame_sequence.frame_shape[1]
    gf_params["minDistance"] = int((num_of_pix / max_corners) ** 0.5)

    init_corners = cv2.goodFeaturesToTrack(image_0, maxCorners=max_corners, **gf_params)

    corners = FrameCorners(
        np.array(range(len(init_corners))),
        np.array(init_corners),
        np.array([block_size] * len(init_corners))
    )
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        i1 = image_to_uint8(image_1)

        flow, status, _ = cv2.calcOpticalFlowPyrLK(i0, i1, np.float32(corners.points), None)
        r_flow, r_status, _ = cv2.calcOpticalFlowPyrLK(i1, i0, flow, None)

        status = status.squeeze().astype(np.bool)
        bumped_corners = ((np.float32(corners.points) - r_flow) ** 2).reshape(-1, 2).max(-1) < 0.25

        corners = FrameCorners(corners.ids, flow, corners.sizes)
        corners = filter_frame_corners(corners, np.logical_and(status, bumped_corners))

        if len(corners.points) < max_corners:
            mask = create_points_mask(i1, corners)

            diff = max_corners - len(corners.points)
            new_corners = cv2.goodFeaturesToTrack(image_1, mask=mask, maxCorners=diff, **gf_params)
            if new_corners is None:
                new_corners = []

            new_id = corners.ids[-1][0] + 1
            new_corners = FrameCorners(
                np.array(range(new_id, new_id + len(new_corners)), dtype=np.int64),
                np.array(new_corners),
                np.array([block_size] * len(new_corners)))

            corners = concat_corners(corners, new_corners)

        builder.set_corners_at_frame(frame, corners)
        i0 = i1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

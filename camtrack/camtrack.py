#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    pose_to_view_mat3x4,
    TriangulationParameters,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

# Good for fox_short
# MAX_REPROJ_ERR = 1
# MIN_TRIANG_ANGLE = 0.2
# MIN_DEPTH = 0.01

MAX_REPROJ_ERR = 5
MIN_TRIANG_ANGLE = 0.01
MIN_DEPTH = 0.001
TRIANG_PARAMS = TriangulationParameters(
    max_reprojection_error=MAX_REPROJ_ERR,
    min_triangulation_angle_deg=MIN_TRIANG_ANGLE,
    min_depth=MIN_DEPTH
)

HALF_FRAME_DIST = 180


def add_points(point_cloud_builder, corner_storage, i, view_mats, intrinsic_mat):
    cloud_changed = False
    for d in range(-HALF_FRAME_DIST, HALF_FRAME_DIST):
        if d == 0 or i + d < 0 or i + d >= len(view_mats) or view_mats[i + d] is None:
            continue
        correspondences = build_correspondences(corner_storage[i + d], corner_storage[i], point_cloud_builder.ids)
        if len(correspondences.ids) > 0:
            points, ids, _ = triangulate_correspondences(correspondences, view_mats[i + d], view_mats[i],
                                                         intrinsic_mat, TRIANG_PARAMS)
            if len(ids) > 0:
                cloud_changed = True
                point_cloud_builder.add_points(ids, points)

    return cloud_changed


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    frames_num = len(rgb_sequence)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    view_mat1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat2 = pose_to_view_mat3x4(known_view_2[1])

    points, ids, _ = triangulate_correspondences(correspondences, view_mat1, view_mat2, intrinsic_mat, TRIANG_PARAMS)

    if len(ids) < 8:
        print("Bad frames: too few correspondences!")
        exit(0)

    view_mats, point_cloud_builder = [None] * frames_num, PointCloudBuilder(ids, points)
    view_mats[known_view_1[0]] = view_mat1
    view_mats[known_view_2[0]] = view_mat2

    updated = True
    pass_num = 0
    while updated:
        updated = False
        pass_num += 1

        for i in range(frames_num):
            if view_mats[i] is not None:
                continue

            corners = corner_storage[i]

            _, ind1, ind2 = np.intersect1d(point_cloud_builder.ids, corners.ids.flatten(), return_indices=True)
            try:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                    point_cloud_builder.points[ind1], corners.points[ind2],
                    intrinsic_mat, distCoeffs=None
                )

                print(f"Pass {pass_num}, frame {i}. Number of inliers == {len(inliers)}")

                if len(inliers) == 0:
                    continue

                view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            except Exception:
                continue

            updated = True

            cloud_changed = add_points(point_cloud_builder, corner_storage, i, view_mats, intrinsic_mat)
            if cloud_changed:
                print(f"Size of point cloud == {len(point_cloud_builder.ids)}")

    first_not_none = next(item for item in view_mats if item is not None)
    if view_mats[0] is None:
        view_mats[0] = first_not_none

    for i in range(frames_num):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )

    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

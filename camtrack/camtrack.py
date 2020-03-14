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
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    _remove_correspondences_with_ids
)

MAX_REPROJ_ERR = 7.9
MIN_TRIANG_ANGLE = 5.2
MIN_DEPTH = 0.3
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


def validate(correspondences, e_inliers):
    _, h_inliers = cv2.findHomography(correspondences.points_1, correspondences.points_2, method=cv2.RANSAC)

    return sum(e_inliers.flatten()) >= sum(h_inliers.flatten())


def select_frames(frames, corner_storage, camera_matrix):
    frame1 = (0, view_mat3x4_to_pose(eye3x4()))
    frame2 = (-1, view_mat3x4_to_pose(eye3x4()))

    mx = 0

    for i in range(1, len(frames)):
        correspondences = build_correspondences(corner_storage[frame1[0]], corner_storage[i])
        if len(correspondences.ids) < 8:
            continue

        E, mask = cv2.findEssentialMat(
            correspondences.points_1, correspondences.points_2, camera_matrix,
            method=cv2.RANSAC
        )

        if mask is None:
            continue

        correspondences = _remove_correspondences_with_ids(correspondences, np.argwhere(mask.flatten() == 0))
        if len(correspondences.ids) < 8 or not validate(correspondences, mask):
            continue

        R1, R2, t = cv2.decomposeEssentialMat(E)
        ps = [Pose(R1.T, R1.T @ t), Pose(R2.T, R2.T @ t), Pose(R1.T, R1.T @ (-t)), Pose(R2.T, R2.T @ (-t))]

        for pose in ps:
            points, _, _ = triangulate_correspondences(
                correspondences,
                pose_to_view_mat3x4(frame1[1]), pose_to_view_mat3x4(pose),
                camera_matrix, TRIANG_PARAMS)

            if len(points) > mx:
                frame2 = (i, pose)
                mx = len(points)

    print(frame1[0], frame2[0])

    return frame1, frame2

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    frames_num = len(rgb_sequence)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = select_frames(rgb_sequence, corner_storage, intrinsic_mat)
        if known_view_2[0] == -1:
            print("Failed to find good starting frames")
            exit(0)

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

                if inliers is None:
                    print(f"Pass {pass_num}, frame {i}. No inliers!")
                    continue

                print(f"Pass {pass_num}, frame {i}. Number of inliers == {len(inliers)}")

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

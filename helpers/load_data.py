import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from preprocess import cgDataset, Options
import os


def transform_pointcloud(xyz_points, rigid_transform):
    xyz_points = np.dot(rigid_transform[:3, :3], xyz_points.T)  # apply rotation
    xyz_points = xyz_points + np.tile(rigid_transform[:3, 3].reshape(3, 1),
                                      (1, xyz_points.shape[1]))  # apply translation
    return xyz_points.T


def backproject_target(curr_pose, cam_target, cam_intr, imsize, center_p=np.array([[0.0370, 0.0213, 0.2462]])):
    pixel_xy = np.zeros((1, 2))
    center_p_w = transform_pointcloud(center_p, cam_target)
    xy, xyzcam = project3Dto2D(center_p_w, curr_pose[:, :], cam_intr)
    pixel_xy[0, :] = xy

    valid_x = np.bitwise_and(pixel_xy[:, 0] > 5, pixel_xy[:, 0] < imsize[0] - 5)
    valid_y = np.bitwise_and(pixel_xy[:, 1] > 5, pixel_xy[:, 1] < imsize[1] - 5)
    invalid = np.bitwise_not(np.bitwise_and(valid_x, valid_y))
    return pixel_xy, invalid


#   xyz_points      - Nx3 float array of 3D points
def project3Dto2D(point_3d, camera_pose, cam_intr):
    xyz_cam = transform_pointcloud(point_3d, invRt(camera_pose))
    xyz_2d = np.dot(cam_intr, xyz_cam.T)
    xyz_2d = xyz_2d.T
    xyz_2d = xyz_2d / np.tile(xyz_2d[:, [2]], [1, 3])

    xyz_2d = np.round(xyz_2d[:, 0:2])
    return xyz_2d, xyz_cam


def invRt(Rt):
    invR = Rt[0:3, 0:3].T
    invT = -1 * np.dot(invR, Rt[0:3, 3])
    invT.shape = (3, 1)
    RtInv = np.concatenate((invR, invT), axis=1)
    RtInv = np.concatenate((RtInv, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return RtInv


##########################################################################################

opt = Options().parse()
opt.folders = []
allfloder = ['000']
opt.dataroot = 'data/'
tool_tip_location = np.array([[0.0370, 0.0213, 0.2462]])  # tool tip location in camera coordinate
cam_intr = np.loadtxt(opt.dataroot + "intrinsics.txt", delimiter=' ')
imsize = np.array([640, 360])

for k in range(len(allfloder)):
    folder = allfloder[k]
    for f in os.scandir(os.path.join(opt.dataroot, folder)):
        if f.is_dir():
            opt.folders.append(folder + '/' + f.name)

dataloader = cgDataset(opt)

while True:
    # load one random grasping sequence
    folder_id, seq_id = dataloader.set_folder_seq_id('random')
    # find starting and end frame (gripper close) for one apporaching trajectory
    [start_frame_id, grasp_frame_id, end_frame_id] = dataloader.get_seq_seg(seq_id)

    # load data
    for frame_id in range(start_frame_id, grasp_frame_id + 5):
        # reading RGB-D image
        data = dataloader.get_rgbd_img(frame_id)
        # reading camera pose
        curr_pose = dataloader.get_campose(frame_id)
        grasp_pose = dataloader.get_campose(grasp_frame_id)

        # visulize the data
        plt.clf()
        plt.imshow(data['color_im'])

        # draw the next 5 frames gripper location
        all_grasp_pixel = np.empty((0, 2))
        for j in range(frame_id + 1, min(grasp_frame_id, frame_id + 5)):
            next_pose = dataloader.get_campose(j)
            pixel_xy, invalid = backproject_target(curr_pose, next_pose, cam_intr, imsize, tool_tip_location)
            all_grasp_pixel = np.vstack((all_grasp_pixel, pixel_xy))
            plt.scatter(pixel_xy[0, 0], pixel_xy[0, 1], c='k', s=40)
            plt.scatter(pixel_xy[0, 0], pixel_xy[0, 1], c='g', s=30)

        # draw the next 5 frames gripper trajectory
        plt.plot(all_grasp_pixel[:, 0], all_grasp_pixel[:, 1], 'g')

        # draw the final grasp point - note that the final grasp point is not always visiable in the current frame.
        pixel_xy, invalid = backproject_target(curr_pose, dataloader.get_campose(grasp_frame_id), cam_intr, imsize,
                                               tool_tip_location)
        plt.scatter(pixel_xy[0, 0], pixel_xy[0, 1], c='k', s=40)
        plt.scatter(pixel_xy[0, 0], pixel_xy[0, 1], c='b', s=30)
        plt.pause(0.02)
    A = input("Enter Y to continue. N to exit: ")
    if A == 'N':
        break

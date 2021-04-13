import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import argparse
import cv2


def read_file_list(file_list):
    with open(file_list, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def check_jpg_png(path):
    npath = path

    if os.path.isfile(path + '.jpg'):
        npath = path + '.jpg'
    elif os.path.isfile(path + '.png'):
        npath = path + '.png'
    else:
        print(path + '.jpg')
        print("file not find" + npath)

    return npath


def makefullpath(directory, full_list, idx):
    pathes = []
    for i in range(len(full_list)):
        pathes.append(os.path.join(directory, full_list[i][idx] + '.color'))
    return pathes


class cgDataset():
    # @staticmethod
    def __init__(self, opt):
        #     self.initialize(opt,folders)
        # def initialize(self, opt,folders):
        self.opt = opt
        self.dataroot = opt.dataroot
        self.folders = opt.folders
        self.numfolder = len(self.folders)
        self.max_depth = opt.max_depth
        self.numoffolder = len(self.folders)
        self.meta = {}
        self.seg = {}
        self.campose = {}
        for i in range(self.numoffolder):
            self.meta[self.folders[i]] = read_file_list(os.path.join(opt.dataroot, self.folders[i], 'meta.txt'))
            self.seg[self.folders[i]] = read_file_list(os.path.join(opt.dataroot, self.folders[i], 'segment.txt'))
            self.campose[self.folders[i]] = np.loadtxt(os.path.join(opt.dataroot, self.folders[i], 'cam_pose.txt'))

        self.currfolder_id = 0
        self.currseq_id = -1

    def set_folder_seq_id(self, sample_type, fid=-1, sid=1):
        if sample_type == 'inorder':
            self.currseq_id = (self.currseq_id + 1)
            if self.currseq_id >= self.numseq():
                self.currseq_id = 0
                self.currfolder_id = (self.currfolder_id + 1) % self.numfolder

        elif sample_type == 'random':
            self.currfolder_id = np.random.randint(self.numfolder)
            self.currseq_id = np.random.randint(self.numseq(self.currfolder_id))
        elif sample_type == 'fix':
            self.currfolder_id = fid
            self.currseq_id = sid

        return self.currfolder_id, self.currseq_id

    def numseq(self, folder_id=-1):
        if folder_id == -1:
            folder_id = self.currfolder_id
        return len(self.seg[self.folders[folder_id]])

    def get_rgbd_img(self, frameId):
        currfolder = self.folders[self.currfolder_id]
        fullpath = os.path.join(self.dataroot, currfolder, self.meta[currfolder][frameId][1])
        color_im = cv2.cvtColor(cv2.imread(check_jpg_png(fullpath + '.color')), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(check_jpg_png(fullpath + '.depth'), -1).astype(
            float) / 10000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > self.max_depth] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = self.campose[currfolder][frameId * 3:frameId * 3 + 3, :]
        return {'color_im': color_im, 'depth_im': depth_im, 'cam_pose': cam_pose}

    def get_campose(self, frameId):
        cam_pose = self.campose[self.folders[self.currfolder_id]][frameId * 3:frameId * 3 + 3, :]
        return cam_pose

    def get_seq_seg(self, seqId):
        seg = np.array(self.seg[self.folders[self.currfolder_id]][seqId])
        start_frameId = int(seg[0]) - 1
        grasp_frameId = int(seg[1]) - 1
        end_frameId = int(seg[2]) - 1
        return start_frameId, grasp_frameId, end_frameId

    def setfolder_seqid(self):
        pass


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', default='metric', help='input video seq length')
        parser.add_argument('--dataroot', default='/media/shurans/My Book/copy_grasp_data/', help='path to images')
        parser.add_argument('--max_depth', type=float, default=1, help='path to images')
        parser.add_argument('--use_metricnet', type=int, default=1)

        parser.add_argument('--phase', default='train', help='input batch size')
        parser.add_argument('--gpu', default='0', help='scale images to this size')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay for adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--momentum', type=float, default=0.9)

        parser.add_argument('--check_point_dir', default='./check_point', help='scale images to this size')
        parser.add_argument('--max_iter', type=int, default=100000)
        parser.add_argument('--modeltype', default='dense_net', help='')
        parser.add_argument('--losstype', default='reg', help='cls or reg')
        parser.add_argument('--datatype', default='rgbnd', help='rgbd rgbnd')

        parser.add_argument('--load_dir', default='', help='')
        parser.add_argument('--scene_type', default='', help='')
        parser.add_argument('--in_real', type=int, default=0, help='')

        parser.add_argument('--continue_train', type=int, default=0, help='')
        parser.add_argument('--enble_6Dof', type=int, default=0, help='')
        parser.add_argument('--load_model', type=int, default=1)
        parser.add_argument('--dofinetune', type=int, default=1, help='')
        parser.add_argument('--log_dir', default='log', help='')
        parser.add_argument('--pt_dir', default='log', help='directory of pretrained model')

        parser.add_argument('--object_type', default='normal', help='')
        parser.add_argument('--one_step', type=int, default=0, help='')
        parser.add_argument('--use_currstate', type=int, default=1, help='')
        parser.add_argument('--discount', type=float, default=0.95, help='')
        parser.add_argument('--gui_enabled', type=int, default=1, help='')
        parser.add_argument('--angle_thre', type=float, default=18, help='')

        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        opt.name = opt.name + opt.modeltype
        opt.fullcheckpoint = opt.check_point_dir + '/' + opt.name + '/'
        opt.log_dir = opt.log_dir + '_' + opt.scene_type
        if not os.path.exists(opt.fullcheckpoint):
            os.makedirs(opt.fullcheckpoint)

        return opt

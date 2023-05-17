# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import os.path as osp
import pickle
import torch
import smplx
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

"""
This scripts imports a pkl file stored SMPL-X parameters and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# SMPL & SMPLX body models
model_path = 'poselib/data/smpl_model'
gender = 'male'
smpl_body_model = smplx.create(
    model_path=model_path,
    model_type='smpl',
    gender=gender,
    batch_size=1
)
# smplx_body_model = smplx.create(
#     model_path=model_path,
#     model_type='smplx',
#     gender=gender,
#     use_pca=False,
#     batch_size=1
# )

smpl_joint_names = [
     'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 
     'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist','L_Hand','R_Hand'
]

parents = smpl_body_model.parents.numpy()[:24]

# calculate joint positions
deta = np.load("poselib/data/cmu_smpl/betas.npy", allow_pickle=True)
betas = torch.tensor(deta[:10], dtype=torch.float32).reshape(1, 10)
# clip = '0000000000'
# input_path = osp.join(dataset_dir, f'{clip}.pkl')
# with open(input_path, 'rb') as f:
#     data = pickle.load(f, encoding='latin1')
#     betas = torch.tensor(data['person00']['betas'][:10], dtype=torch.float32).reshape(1,10)
joints_pos = smpl_body_model(betas=betas).joints[0, :].detach().numpy()

# extract SMPL joints from SMPLX joints
joints_to_use = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
)
joints_pos = joints_pos[joints_to_use]


# create SkeletonTree
local_joints_pos = joints_pos.copy()
for jid in range(len(parents)):
    pid = parents[jid]
    if pid != -1:
        local_joints_pos[jid] = joints_pos[jid] - joints_pos[pid]
# for jid in range(len(parents)):
#     pid = parents[jid]
#     if pid != -1:
#         local_joints_pos[jid] = joints_pos[jid] - joints_pos[pid]
# for jid in range(len(parents)):
#     local_joints_pos[jid][[1,2]] =  local_joints_pos[jid][[2,1]]
skeleton = SkeletonTree(node_names=smpl_joint_names, parent_indices=torch.LongTensor(parents), local_translation=torch.FloatTensor(local_joints_pos))

# generate zero rotation pose
zero_pose = SkeletonState.zero_pose(skeleton)

# save and visualize T-pose
zero_pose.to_file("poselib/data/tpose/smpl_tpose.npy")
plot_skeleton_state(zero_pose)

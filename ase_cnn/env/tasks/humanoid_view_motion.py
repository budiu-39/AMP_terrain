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

import torch

from isaacgym import gymtorch
import numpy as np

from env.tasks.humanoid_amp import HumanoidAMP
from isaacgym import gymtorch
from isaacgym.terrain_utils import *
from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import *


class HumanoidViewMotion(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self.create_terrains()
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = torch.zeros_like(self.actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        return

    def create_terrains(self):
        self._num_terrains = self.num_envs
        num_per_row = np.sqrt(self.num_envs)

        terrain_width = 10.
        terrain_length = 10.

        horizontal_scale = 0.04  # [m]
        self._horizontal_scale = horizontal_scale
        vertical_scale = 0.02  # [m]
        self._vertical_scale = vertical_scale
        num_rows = int(terrain_width / horizontal_scale)  # 高度图为的数组为num_row*num_cols  保存在terrain里面， 乘horizontal_scale以后是真实的大小
        num_cols = int(terrain_length / horizontal_scale)  # vertical_scale是高度的比例，其中terrain生成函数里面的输入就是真实的大小，如果scale（像素的）的大小比真实大小还大就会发生用斜坡补充的情况
        def new_sub_terrain():  return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                                                 horizontal_scale=horizontal_scale)

        self.obs_heightfield = self.stage_stairs_terrain(new_sub_terrain(), step_width=0.32, step_height=0.16)  # 这个用来作网络输入
        self.heightfield = np.zeros((int(num_per_row) * num_rows, (int(self._num_terrains/int(num_per_row))+1) * num_cols))  # 这个用来记录环境的大env
        self.obs_heightfield_cuda = torch.tensor(self.obs_heightfield, device=self.device)
        for i in range(int(num_per_row)):
            for j in range(int(self._num_terrains/int(num_per_row))+1):
                self.heightfield[i * num_rows: (i+1) * num_rows, j * num_cols: (j+1) * num_cols] =  self.obs_heightfield
        vertices, triangles = convert_heightfield_to_trimesh(self.heightfield, horizontal_scale=horizontal_scale,
                                                             vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = -0.01
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
        return


    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return


    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_actors(self, env_ids):
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated
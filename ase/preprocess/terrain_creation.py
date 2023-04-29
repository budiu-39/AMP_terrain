"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Terrain examples
-------------------------
Demonstrates the use terrain meshes.
Press 'R' to reset the  simulation
"""

import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os
import trimesh
from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt

# import os
# path = os.environ["PATH"]
# path = r'/media/srtp/新加卷/SRTP_MotionGeneration/v-hacd-master/app/build'
# os.environ["PATH"] = os.environ["PATH"]+';'+path
# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# load ball asset
# asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
cur_path = os.path.abspath(os.path.dirname(__file__))   # 获取当前文件的目录
proj_path = cur_path[:cur_path.find('preprocess')]
asset_root =os.path.join(proj_path)
asset_file = "data/assets/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# set up the env grid
num_envs = 10
num_per_row = 2
env_spacing = 0.56
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 1.
pose.p.x = 3.

envs = []
# set random seed
np.random.seed(17)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# create all available terrain types
terrain_width = 10.
terrain_length = 10.
horizontal_scale = 0.1  # [m]
vertical_scale = 0.1  # [m]
num_rows = int(terrain_width/horizontal_scale)   #高度图为的数组为num_row*num_cols  保存在terrain里面， 乘horizontal_scale以后是真实的大小
num_cols = int(terrain_length/horizontal_scale)  #vertical_scale是高度的比例，其中terrain生成函数里面的输入就是真实的大小，如果scale（像素的）的大小比真实大小还大就会发生用斜坡补充的情况
def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
terrain = pyramid_stairs_terrain(new_sub_terrain(), step_width=1.5, step_height=0.3).height_field_raw

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(terrain, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)


# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 15)
cam_target = gymapi.Vec3(0, 0, 10)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
tri_terrain = trimesh.Trimesh(vertices=vertices, faces=triangles)
# tri_terrain.show()
trimesh.exchange.export.export_mesh(tri_terrain, "ase/data/assets/terrain.obj")
# trimesh.exchange.urdf.export_urdf(tri_terrain, "ase/data/assets", scale=1.0, color=None)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

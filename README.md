# AMP_terrain

This is a project based on ASE from Xue bin Peng. It is my graduation project. I mainly use the AMP part from ASE project, and add my own code.

### AMP

We also provide an implementation of Adversarial Motion Priors (https://xbpeng.github.io/projects/AMP/index.html).
A model can be trained to imitate a given reference motion using the following command:
```
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --headless --num_envs 32

python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --num_envs 32
```
The trained model can then be tested with:
```
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint [path_to_amp_checkpoint]
```
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/amp_humanoid_jog.npy  --checkpoint /media/srtp/新加卷/SRTP_MotionGeneration/demo/run/nn/Humanoid.pth 



 python ase/run.py --task HumanoidLocation --num_envs 16 --cfg_env ase/data/cfg/humanoid_location.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --headless

带目标
python ase/run.py --test --task HumanoidLocation --num_envs 16 --cfg_env ase/data/cfg/humanoid_location.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --checkpoint /media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/output/Humanoid_31-22-06-06/nn/Humanoid.pth 

带地形
python ase/run.py --test --task HumanoidLocationScene --num_envs 16 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --checkpoint /media/srtp/新加卷/SRTP_MotionGeneration/demo/walk_task_scene/nn/Humanoid.pth 


python ase/run.py --task HumanoidLocationScene --num_envs 256 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/amp_humanoid_walk.npy --headless


上楼梯 带地形
python ase/run.py --task HumanoidLocationScene --num_envs 1024 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/style_1/walk_stair.yaml --headless

python ase/run.py --test --task HumanoidLocationScene --num_envs 16 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/style_1/walk_stair.yaml --checkpoint /media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/output/Humanoid_24-23-03-06/nn/Humanoid.pth

python ase/run.py --test --task HumanoidAMPScene --num_envs 16 --cfg_env ase/data/cfg/humanoid_amp_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/style_1/up_down_stair.npy  --checkpoint /media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/output/Humanoid_28-12-50-30/nn/Humanoid.pth 

python ase/run.py --task HumanoidLocationScene --num_envs 512 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/style_1/amp_humanoid_walk.npy --headless

ffmpeg -i input_motion.webm -crf 17 -c:v libx264 input_motion.mp4
&nbsp;

&nbsp;

### Motion Data

Motion clips are located in `ase/data/motions/`. Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/data/motions/style_1/up_down_stair.npy
```
`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.


This motion data is provided courtesy of Reallusion, strictly for noncommercial use. The original motion data is available at:

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-stunts

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-moves


If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `ase/poselib/retarget_motion.py`.

# AMP_terrain

This is a project based on ASE from Xue bin Peng. It is my graduation project. I mainly use the AMP part from ASE project, and add my own code.

Codebase: https://github.com/nv-tlabs/ASE

In file ase, I add terrrain in environment. 
In file ase_cnn, I change the structure of network and add cnn layers. But it doesn't work very well.

Better condition comes soon (Hopefully...

### to run AMP with terrain (in file ase
```
python ase/run.py --task HumanoidLocationScene --num_envs 512 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_task.yaml --motion_file ase/data/motions/style_1/amp_humanoid_walk.npy --headless
```

&nbsp;

&nbsp;

### to run AMP with terrain and cnn layers (in file ase_cnn
```
Python run.py --task HumanoidLocationScene --num_envs 256 --cfg_env ase_cnn/data/cfg/humanoid_location_scene.yaml --cfg_train ase_cnn/data/cfg/train/rlg/humanoid_terrain.yaml --motion_file ase_cnn/data/motions/style_1/walk_stair.yaml --headless
```





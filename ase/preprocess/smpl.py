import os
import smplx
import torch
import pickle
import trimesh


input_path = "/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0052/params/" 
output_path = "/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0052/mesh/" 
model_path = "/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/smpl_model/"
gender = "male"
matrix=[[1,0,0,0],[0,1,0,0.10],[0,0,1,0],[0,0,0,1]]

os.makedirs(output_path, exist_ok=True)
body_model = smplx.create(model_path=model_path,
                             model_type='smpl',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)

for info in os.listdir(input_path): 
    info_pkl = os.path.join(input_path,info) #将路径与文件名结合起来就是每个文件的完整路径  
    with open(info_pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # fr = data['mocap_framerate']
        full_poses = torch.tensor(data['person00']['body_pose'], dtype=torch.float32)
        betas = torch.tensor(data['person00']['betas'][:10], dtype=torch.float32).reshape(1,10)
        full_trans = torch.tensor( data['person00']['transl'], dtype=torch.float32)
        # print("Number of frames is {}".format(full_poses.shape[0]))

    global_orient = torch.tensor(data['person00']['global_orient'], dtype=torch.float32).reshape(1,-1)
    body_pose = full_poses[:].reshape(1,-1)
    transl = full_trans[:].reshape(1,-1)
    output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
    m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
    m = m.apply_transform(matrix) 
    path=output_path + info.split('.')[0] + '.obj'
    m.export(path)

    

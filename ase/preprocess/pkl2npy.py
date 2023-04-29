import numpy as np
import smplx  
import pandas as pd
import os
import collections
import torch

from scipy.spatial.transform import Rotation as R



# test = pd.read_pickle('/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/poselib/data')
# # print(test)
# # np.save('smpl_tpose.npy',test)
# doc = open('smpl_tpose.txt', 'a')
# print(test, file=doc)

# motion_file='/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0000/params'
# local_rotation=[]
# root_translation=[]
# arrr={}
# arrt={}
# source_motion=collections.OrderedDict()
# files = os.listdir(motion_file)
# files.sort()
# print(files)
# for pkl in files:
#     pkl = os.path.join(motion_file,pkl)
#     motion = np.load(pkl,allow_pickle=True)
#     rot=np.array(motion['person00']['pose'])
#     rot=rot.reshape(-1,3)
#     r = R.from_rotvec(rot)      
#     local_rotation.append(r.as_quat())
#     root_translation.append(motion['person00']['transl'])

r = R.from_rotvec([0,0,0])
print(r.as_quat()) 

# arrr['arr'] = np.array(local_rotation)
# arrr['context'] = {'dtype':'float32'}
# source_motion['rotation'] = arrr
# arrt['arr'] = np.array(root_translation) 
# arrt['context'] = {'dtype':'float32'}
# source_motion['root_translation'] = arrt
# source_motion['is_local']=True,
# source_motion['fps']= 30
# source_motion['__name__']='SkeletonMotion'


# np.save('smpl_motion.npy',source_motion)

# data=np.load("/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/poselib/data/smpl_tpose.npy", allow_pickle = True)
# # print('data:\n',data)
# doc = open('smpl_tpose.txt', 'a')
# d=data.item()
# # len(d['rotation']['arr'])
# print(data, file=doc)

# pkl_motion = np.load('/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/smpl_tpose.npy', allow_pickle = True)
# print(pkl_motion)
# data=pkl_motion['person00']
# type(data)
# #print(amp_motion['person00'])
# print("pkl")
# for i,j in data.items():
#     print(i+':', end='')
#     if type(j)!=float and type(j)!=int :
#         print(j.shape)
# print(" ")

# cmu_motion = np.load('/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/poselib/data/cmu_tpose.npy', allow_pickle = True)
#data=amp_motion['person00']
#print(amp_motion['person00'])
# print(cmu_motion)
# data = cmu_motion.item()
# for i,j in data.items():
    
#     if type(j) == dict:
#         print(i+':', end=' ')
#         # print(j)
#         for x,y in j.items():
#             if type(y) != dict:
#                 print(y.shape)

    #print(type(j))
    #print(' ')  
    #print(cmu_motion['i'])
    #if type(j)!=float and type(j)!=int :
        #print(j.shape)


print(' ')


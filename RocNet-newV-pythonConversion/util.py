import os
from argparse import ArgumentParser
import easydict

# + Nov 2023
import numpy as np
import torch

def get_args():
#     parser = ArgumentParser(description='grass_pytorch')
#     parser.add_argument('--box_code_size', type=int, default=12)
#     parser.add_argument('--feature_size', type=int, default=80)
#     parser.add_argument('--hidden_size', type=int, default=200)
#     parser.add_argument('--symmetry_size', type=int, default=8)
#     parser.add_argument('--max_box_num', type=int, default=30)
#     parser.add_argument('--max_sym_num', type=int, default=10)

#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch_size', type=int, default=123)
#     parser.add_argument('--show_log_every', type=int, default=3)
#     parser.add_argument('--save_log', action='store_true', default=False)
#     parser.add_argument('--save_log_every', type=int, default=3)
#     parser.add_argument('--save_snapshot', action='store_true', default=False)
#     parser.add_argument('--save_snapshot_every', type=int, default=5)
#     parser.add_argument('--no_plot', action='store_true', default=False)
#     parser.add_argument('--lr', type=float, default=.001)
#     parser.add_argument('--lr_decay_by', type=float, default=1)
#     parser.add_argument('--lr_decay_every', type=float, default=1)

#     parser.add_argument('--no_cuda', action='store_true', default=False)
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--data_path', type=str, default='data')
#     parser.add_argument('--save_path', type=str, default='models')
#     parser.add_argument('--resume_snapshot', type=str, default='')
#     args = parser.parse_args()
#     return args

    args = easydict.EasyDict({
        "box_code_size": 512,
        "feature_size": 80,
        "hidden_size": 200,
        "max_box_num": 6126,
        "epochs": 10,
        "batch_size": 10,
        "show_log_every": 1,
        "save_log": True,
        "save_log_every": 3,
        "save_snapshot": True,
        "save_snapshot_every": 5,
        "save_snapshot":'snapshot',
        "no_plot": False,
        "lr":0.001,
        #"lr": 0.1,
        "lr_decay_by":1,
        "lr_decay_every":1,
        "no_cuda": True,
        "gpu":1,
        "data_path":'data',
        "save_path":'models',
        "resume_snapshot":"",
        "leaf_code_size":1
    })
    return args
    
    
    
# added by SBN. October 2023    
def get_feas_vox3d_v5(vox, k):
    '''
       **converted Matlab function. 
       Date: October 2023.
    '''


    def get_labels_fill_feas(vox, k):
        n =  vox.shape[0]
        if n<k:
            print('dim must be larger than k')
            return

        # all non-zero

        if torch.all(vox!=0):
            fea_global.append(vox)
            label = torch.tensor([0])
            return  label

        # all zero
        if torch.all(vox==0):
            fea_global.append(vox)
            label = torch.tensor([1])
            return  label

        if n==k:
            fea_global.append(vox)
            label = torch.tensor([2])
            return  label


        l1 = get_labels_fill_feas(vox[:n//2  , :n//2  , :n//2],k);
        l2 = get_labels_fill_feas(vox[n//2: , :n//2  , :n//2],k);
        l3 = get_labels_fill_feas(vox[:n//2  , n//2: , :n//2],k);
        l4 = get_labels_fill_feas(vox[n//2: , n//2: , :n//2],k);

        l5 = get_labels_fill_feas(vox[:n//2  , :n//2 , n//2:],k);
        l6 = get_labels_fill_feas(vox[n//2: , :n//2 , n//2:],k);
        l7 = get_labels_fill_feas(vox[:n//2  , n//2:, n//2:],k);
        l8 = get_labels_fill_feas(vox[n//2: , n//2:, n//2:],k);

#         print('labels: ', l1,l2,l3,l4,  l5,l6,l7,l8)
        label = torch.cat((l1,l2,l3,l4,  l5,l6,l7,l8,  torch.tensor([3])), axis=0)

        return label


    fea_global = []
    labels = get_labels_fill_feas(vox, k)
    
    # + Nov 
    fea_global1 = torch.empty((vox.numel()//k**3, k, k, k))
    for idx, fea in enumerate(fea_global):
    	#print(fea)
    	fea_global1[idx] = fea.reshape(fea.shape[0], fea.shape[1], fea.shape[2])
    
    # - May 8
    # return fea_global, labels
    # + May
    return fea_global1, labels.reshape(1,-1)

def get_tree_vox_v2(feas_all, label, vox_size):
    """
    **converted Matlab function. 
    Generate a tree of voxels based on the given features and labels.

    Args:
    feas_all (array): Array of features
    label (array): Array of labels
    vox_size (int): Size of the voxel

    Returns:
    vox (array): Voxel representation
    label (array): Updated label
    feas_all (array): Updated features
    
    Date: Nov 2023
    """

    vox = torch.zeros((vox_size, vox_size, vox_size))

    if label[-1] == 0:
        vox = 1
        label = label[:-1]
        feas_all = feas_all[:-1, :, :, :]
        return vox, label, feas_all
    elif label[-1] == 1:
        vox = 0
        label = label[:-1]
        feas_all = feas_all[:-1, :, :, :]
        return vox, label, feas_all
    elif label[-1] == 2:
        #print(feas_all[-1, :, :, :].shape)
        vox = feas_all[-1, :, :, :]#, (vox_size, vox_size, vox_size))
        label = label[:-1]
        feas_all = feas_all[:-1, :, :, :]
        return vox, label, feas_all
    else:
        #print(feas_all[-1, :, :, :].shape)
        label = label[:-1]
        vox1, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox2, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox3, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox4, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox5, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox6, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox7, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)
        vox8, label, feas_all = get_tree_vox_v2(feas_all, label, vox_size // 2)

        vox[:vox_size // 2, :vox_size // 2, :vox_size // 2] = vox8
        vox[vox_size // 2:, :vox_size // 2, :vox_size // 2] = vox7
        vox[:vox_size // 2, vox_size // 2:, :vox_size // 2] = vox6
        vox[vox_size // 2:, vox_size // 2:, :vox_size // 2] = vox5
        vox[:vox_size // 2, :vox_size // 2, vox_size // 2:] = vox4
        vox[vox_size // 2:, :vox_size // 2, vox_size // 2:] = vox3
        vox[:vox_size // 2, vox_size // 2:, vox_size // 2:] = vox2
        vox[vox_size // 2:, vox_size // 2:, vox_size // 2:] = vox1

    return vox, label, feas_all	    

def cube_voxel(indices, cube_size):
    '''converts a matrix of indecies (2d np.ndarray) showing the occupied voxels 
        into a standard cubic n*n*n by scaling.
    
    indeces  : np.ndarray: This shows the occupancy
                            where row values shows that the acutal shapee's space with 
                            that index is occupied.
                
    cube_size: int       : can be 32, 64, 128 etc. for convenience use values 
                            of power 2.
    '''
    cube_size = cube_size-1 #(ex. 256-->max will be 255)
    coef_cubicize = (indices.max(axis=0)-indices.min(axis=0))/cube_size

    cube_voxels = (indices-indices.min(axis=0))//coef_cubicize
    cube_voxels = cube_voxels.astype(int)
    return cube_voxels
    
def read_sample_cube(file_name, cube_size=256):
    '''This file_name contains all the indeces (non-zero) indeces inside a cube.
    This input should already be voxelized into a cube (not a random point cloud file
    with floating point values).
    After receviing this input, this function will create an occupancy matrix and return
    the cube containing 0s and 1s for un/occupied voxels. The assumption is that all
    the intended are of cube_size=256
    Date: November 2023
    '''
    vox = np.zeros((256,256,256))
    pointcloud = []       
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            xyz = line.strip().split(' ')
            vox[int(float(xyz[0])),int(float(xyz[1])),int(float(xyz[2]))] = 1
            pointcloud.append([int(float(xyz[0])),int(float(xyz[1])),int(float(xyz[2]))])
        
    pointcloud = np.array(pointcloud, dtype=np.int16) 

    indices = pointcloud

    #cube_size = 256
    occupancy_cube_vox = np.zeros((cube_size, cube_size, cube_size), dtype=np.int8) # 0 or 1 in n*n*n 3d matrix
    std_indices = cube_voxel(indices, cube_size)
    std_indices[0]
    # occupancy_cube_vox[std_indices]=1
    for index in std_indices:
        occupancy_cube_vox[index[0], index[1], index[2]]=1  
        
    return occupancy_cube_vox      
	    

	    

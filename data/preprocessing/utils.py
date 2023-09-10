__author__ = 'SBN'
__date__ = 'August 2023'


import numpy as np
from scipy.io import savemat
    
def save_xyz(voxel, file_name):
    vox_size = voxel.shape
    with open(file_name, 'w') as f:
        for i in range(vox_size[0]):
            for j in range(vox_size[1]):
                for k in range(vox_size[2]):
                    if voxel[i][j][k]>0:
                        f.write(f'{i} {j} {k}\n')        
    
    print ("voxel cube data is stored at ", file_name)                    


def save_mat(voxel, file_name):
    '''sutiable for matlab .mat files ...'''
    savemat(file_name, {"vox":voxel})
    print ("voxel cube data is stored at ", file_name)                    
    

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
    
def read_point_cloud(point_cloud_path):
    '''read xyz file file and returns the points poistions.
    The indices of points are returned'''
    pointcloud = []
    with open(point_cloud_path) as f:
        lines = f.readlines()
        for line in lines:

            xyz = line.strip().split(' ')
            #vox[int(float(xyz[0])),int(float(xyz[1])),int(float(xyz[2]))] = 1
            pointcloud.append([float(xyz[0]),float(xyz[1]),float(xyz[2])])

    pointcloud = np.array(pointcloud)
    return pointcloud

def build_cube(indices, cube_size):
    '''getting indices of a point cloud and cube_size, it builds
    a voxel cube for the given indices. 
    
    
    ** Sometimes one can use this for integer
    conversion of already cubic pointclouds. Such pracrices are due to savings in meshlab
    that leads to positions like 251.0000 that is not suitable for indexing. using this function 
    can help doing that conversion into 251 integer.
    '''
    vox = np.zeros((cube_size,cube_size,cube_size), dtype=np.int8)
    for point_index in indices:
        vox[point_index[0], point_index[1], point_index[2]] = 1
    return vox    
        

	    
if __name__ == "__main__":
	pass

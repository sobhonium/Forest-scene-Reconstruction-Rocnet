# RocNet python

This folder is totally changed from the origianl one and that is why I separated this one totally from the rest. 
Since some backbone of the orignal code was based on ```.mat``` files I added/modified functions and files
which changed the original ones into something is not working with the former code base. So, that is why I put all of them here again even if I repeat some of them.

First, Matlab functions are reimplemented in python codes then all the setup changed to read files from local folders.
Previously, I needed to go back and forth from/to Matlab to convert pointcloud cubic data into ```.mat``` and bring them to local directories
folders manually. After training the model for test I also needed to go to Matlab again and convert the reconsucted ones
there. Now, I only read/generate/construct data in this subproject in a fully python based setup.


```/MATLAB``` folder is deleted in this subproject (since not needed anymore). ```get_tree_vox.m``` and ```get_feas_vox.m``` did the enitre convesion and reconstruction in the intial project. These functions are implemented in python and placed in ```util.py``` file.

# Instruction 

- To use this Rocnet version, first you need to put input sample data in ```data/forest/``` folder with enumeration like ```1.xyz```, ```2.xyz```.
The sample data (e.g. ```1.xyz```) is actually a pointcloud containing all the occupancy grid coordinates of the sample data. As an example, if at coordinate ```(1,2,3)``` (in ```1.xyz``` file) there is a non-zero voxel representing a point, there would be a row in ```1.xyz``` pointcloud file like ```1 2 3``` showing this. This will help build the cube afterwards. Since the indices (coordinates) in the cube box are integer all these values must be integer. 

- ```train-256-32-Forest.ipynb``` file to test and work on forest scence training/reconstruction. In this notebook, like before,  ```256*256*256``` cubic box data are trained with the depth of ```32``` in Octree setting.

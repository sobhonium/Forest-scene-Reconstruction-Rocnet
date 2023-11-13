# RocNet python

This folder contains another version of RocNet project and works independantly from the other files put in the root folder.
This project is forked and totally changed from the origianl one and that is why it's separated from the rest. 
Since some backbone of the orignal code was based on MATLAB implementation (```.mat``` files) I added/modified functions and files
which changed the original ones into something that won't work with the former code base. So, that is the reason it's put in here even if I repeat some functions or files. So, this folder can be detached from the root folder and work on its own without any issues.

## Motivation and Change

 The intention was to use/put a pointcloud file containing occupancy grid info of a cube for
showing the cooridinates of values where they are ```1```. We wanted a project version to put this pointcloud into
 ```data/forest/``` folder and replace the former one which used a bunch of ```.mat``` files (```fea_data.mat```,
```label.mat```, and ```op.mat``` for each sample ). Worse, in the pervious project version (i.e. the Matlab version), the conversion, permutes, and so forth very
 crazy and time consuming, and more importantly, the readability and maintainability were not up to the job.

To tackle this, first, Matlab functions are reimplemented in python codes then all the setup changed to read files from local folders.
Previously, I needed to go back and forth from/to Matlab to convert cubic data into ```.mat``` and bring them to local directories
folders manually. After training the model for test I also needed to go to Matlab again and convert the reconsucted ones
there. Now, I only read/generate/construct data in this subproject in a fully python based setup.


```/MATLAB``` folder is deleted in this subproject (since not needed anymore). ```get_tree_vox.m``` and ```get_feas_vox.m``` did the enitre convesion and reconstruction in the intial project. These functions are implemented in python and placed in ```util.py``` file.

# Instruction to Use


- To use this Rocnet version, first you need to put input sample data in ```data/forest/``` folder with enumeration like ```1.xyz```, ```2.xyz```.
The sample data (e.g. ```1.xyz```) is actually a pointcloud containing all the occupancy grid coordinates of cube data. As an example, if at coordinate ```(1,2,3)``` (in ```1.xyz``` file) there is a non-zero voxel representing a point, there would be a row in ```1.xyz``` pointcloud file like ```1 2 3``` showing this. This will help build the cube afterwards. Since the indices (coordinates) in the cube box are integer all these values must be integer. 

- ```train-256-32-Forest.ipynb``` file to test and work on forest scence training/reconstruction. In this notebook, like before,  ```256*256*256``` cubic box data are trained with the depth of ```32``` in Octree setting.

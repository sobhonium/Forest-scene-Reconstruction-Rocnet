# RocNet python

This folder is totally changed from the origianl one and that is why i separate this one totally from the rest. 
Since some backbone of the orignal code was based on .mat files I added/modified functions and files
which totally changed the original one. So, that is why I put all of them here again.

First, matlab functions are reimplemented in python codes then all the setup changed to read files from local folders.
Previously I need to go back and forth from/to Matlab to convert pointcloud cubic data into .mat and bring them to local
folders manually. After training the model for test I also needed to go to Matlab again and convert the reconsucted ones
there. Now, I only read/generate/construct data in this subproject.\


/MATLAB folder is not needed now but I'm keeping it to keep the structure of project (to remember things it's good to be kept). 


Start point: ```train_nb-128-32-Forest.ipynb``` file.

# RocNet python

This folder is totally changed from the origianl one and that is why I separated this one totally from the rest. 
Since some backbone of the orignal code was based on ```.mat``` files I added/modified functions and files
which changed the original ones into something is not working with the former code base. So, that is why I put all of them here again even if I repeat some of them.

First, Matlab functions are reimplemented in python codes then all the setup changed to read files from local folders.
Previously, I needed to go back and forth from/to Matlab to convert pointcloud cubic data into ```.mat``` and bring them to local directories
folders manually. After training the model for test I also needed to go to Matlab again and convert the reconsucted ones
there. Now, I only read/generate/construct data in this subproject in a fully python based setup.\


/MATLAB folder is not needed now in this sufolder (not needed any more), but I'm keeping it to keep the structure of project (to remember things it's good to be kept) as before. 


Start point: ```train_nb-128-32-Forest.ipynb``` file to test and work on forest scence training/reconstruction.

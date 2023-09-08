# Data Preprocessing and preparation
The purpose of preprocssing is to prepare data into a standard cubic form to be readable and understable for the algorith that wants to convert the data into an Octree. So the data should be ```32*32*32```, ```64*64*64```, ...

So this part includes three tasks:
- Reading .las or .laz data and convert it to a voxel-based equivalance data sample (cubic).
- Cleaning the data since the .las data provided is not perfect and some leaves are needed to be deleted.
- Convert the obtained and cleand the into a ```.mat``` file to let the training model read it. The Rocnet turend out that it works only ```.mat``` files.



## Reading .laz 

## Voxelization

## .Mat conversion
After Voxelization of data they need to be in  .mat format. So, as explained, for dataset creation for forest project:

- first I need to have a descent .mat file represeting a tree voxel and save dictionary 'vox' containing 256*256*256 data i.e. dictionary of {"vox", <voxel_name>} in python script in notebooks/DeepLearning_practice/notebooks/volumetric/pointcloud/pcd_read_write.ipynb file. The function that should be run in this notebook is save_mat(occupancy_cube_vox, 'output.mat'), where a voxel shape is enitrely saved in a .mat file.

- then I should move to to matlab import and load the imported  .mat file,  extract voxel, and save it 'vox'  and run get_feas_vox(vox,k) :

>> load output.mat
>> [feas_all,label]=get_feas_vox(vox,256);

- By now, I have returned [feas_all,label]=get_feas_vox(vox,k)  
(I guess Juncheng meant  ops instead of label (run ops=label to assign). 

- Then
>> ops=label ;
>> save ("op_data1.mat", "ops");
>> fea_all = feas_all;
>> save ("fea_data1.mat", "fea_all");

also I need label which is the data label (toilet=0, Desk=1,... in Modelnet10) in a 1*1 matrix and save it as 'labels':
>> labels = 0 
>> save ("label_data1.mat", "labels")

I selected 0 as I only have one class right now. It can be other numbers based on the class number of the sample'sinterest.


- The saved files should be put in the ./data/forest and they create only one sample. for the other samples I should do the same and go and save 3 files and come back and put them in this folder.

- In Juncheng notebook, I should add some changes from ./data/train_1 to ./data/forest (already done with a copy of its notebook and did these changes in that notebook).

I'm not quite sure about this configuration (saving and loading in matalb) but it is actually the closest to my understanding... (maybe I should ask).

Then I should go to the notebook provided and use the  folder name for that from data/train_1 --->./data/forest (already done)

* If you have trouble understanding the shapes see one example from ./data/test_1 and load them on matlab to see their
size and names inside them and save similarly.

* also be sure about the file's name since Juncheng is reading files name with these format mentioned and only number at the end of them change. 

* Sometimes you need to zip fea_data1.mat and download it to your local folder and use it. This should be done since the size of fea_data1.mat is changing when I downlod it. Don't know why!

## Next
The data put in folder ```./data/forest``` is ready to be trained with a model. The notebook []() can be run and the training model can be obtained.
After traing the reconstruction of data can be done. For reconstruction, another step for conversion to/from .mat files are needed...

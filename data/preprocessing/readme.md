# Data Preprocessing and preparation
The purpose of preprocssing is to prepare data into a standard cubic form to be readable and understable for the algorithm that  converts the data into an Octree. So the 3d data should be of cubic size ```32*32*32```, ```64*64*64```, ...

So this part includes three tasks:
- Reading ```.las``` or ```.laz``` data and convert it to a voxel-based equivalent data sample (cubic).
- Cleaning the data since the ```.las``` data provided is not perfect and some leaves are needed to be deleted.
- Convert the obtained and cleand data the into a ```.mat``` file to let the training model read it. The Rocnet turned out to be working only with ```.mat``` files.



## Reading .laz 


The ```.laz``` files are not readable until you convert them via Cloud Compare software. You can convert them into ```.pcd``` (point cloud data type) or ```.las``` files. After converting the lidar ```.laz``` data into ```.las``` you can read ```.las``` data through [this notebook](lidar-las-reading.ipynb). To work with ```laspy``` module your python is needed to be <3.9. 


(recommended) If you convert it to a ```.pcd``` file format, just read and follow [this notebook](pcd_read_write.ipynb).


## Voxelization
For voxelization I recommend using 
[this notebook](pcd_read_write.ipynb) and save the output file with ```.mat``` file extention (if you are using the mentioned notebook, saving to ```.mat``` is provided right in the last cells of the notebook). Let's say the name of the file now is ```output.mat```.

**Note:** Sometimes when data is large, reading through ```.pcd``` was not possible. I used a mixture of las reading and voxelization to do so in [this notebook](voxelize_large_las.ipynb).

**Note:** For processing .las or .laz data you need a machine with high RAM mem. Otherwise, converting with Cloud Compare software, your machine would crash!


## Cleaning and pruning
Sometimes data needs to be cleaned (leaves to be removed) or noises be deleted. For that I recommend using [this](prune-leaves.ipynb) or [this](morphology-dilation.ipynb) to apply morpholoy or others to clean it. Sometimes, however, manually detecting and removing noises are needed. 

In some practices you can use classifiers to help you fasten the cleaning process. I used classifer  ```otira_bedrocksemi.prm``` and use it in Cloud Compre to help cleaning the data. This classifer is trained on leaves and trunks and can be helpful. Remember that this classifier is only a tool and still you need to augment the file manually...

**Note:** The large ```.laz``` or ```.las``` data could not be upload to this repo. I already did pcd conversion, voxelization, cleaning and pruning on one sample data and put them in ```./lidar data/forest-sence-sample.pcd```, ```./lidar data/output-dilated-and-manually-rescaled.xyz``` and ```./lidar data/output-dilated-and-manually-rescaled.mat```.


## Octree data structure conversion
After Voxelization into an output.mat file: 

- you should move to MATLAB and import and load the  .mat file,  extract voxel, and save it 'vox'  and run get_feas_vox(vox,k) :

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

I selected 0 as I only have one class right now. It can be other numbers based on the class number of the sample of interest.


- The saved files ```op_data1.mat, label_data1.mat, fea_data1.mat``` should be put in the ```./data/forest``` and they create/represent only one sample. For the other samples you should do the same and go and save 3 files and come back and put them in this folder again.

- In Juncheng notebook, I should add some changes from ./data/train_1 to ./data/forest (already done with a copy of its notebook and did these changes in that notebook). I already change this in here and no need to change anything right now.

I'm not quite sure about this configuration (saving and loading in matalb) but it is actually the closest to my understanding... (maybe I should ask).

Then I should go to the training [notebook](../../train_nb-128-32-Forest.ipynb) provided and change the  folder name for that from ```data/train_1 --->./data/forest``` (already done)

* also be sure about the file's name since Juncheng is reading files name with these format mentioned and only number at the end of them changed: ```op_data1, op_data2, op_data3,...```

* Sometimes you need to zip fea_data1.mat and download it to your local folder and use it (if you're using MATLAB web)! This should be done since the size of fea_data1.mat is changing when I downlod it. Don't know why!!!

* Now, you are good to train the Rocnet Model (on the sample data you provided) by running training [notebook](../../train_nb-128-32-Forest.ipynb).

## Next
The data put in folder ```./data/forest``` is ready to be trained with a model. The notebook []() can be run and the training model can be obtained.
After traing the reconstruction of data can be done. For reconstruction, another step for conversion to/from .mat files are needed...

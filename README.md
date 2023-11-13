# Forest scene Reconstruction with Rocnet (Octree-based nn)
![image](./data/images/img1.png)


The following instrcutions are tested on conda 23.7.2


To run it you can use a conda exporeted Env.yml file and directly create a env for training:

```conda env create -n <desired-env-name> --file ENV.yml```

```pip install -r requirements.txt```

or to do this automatically just run  


```./install.sh```





Then you should open [the notebook](train_nb-256-32-Forest.ipynb) to do the training.
The notebook is run on sample(s) in ```./data/forest/``` folder (this folder contains datasets). To add/replace data you should convert your sample data into an Octree-based samples. To do so, I recommend to read [this](./data/preprocessing/readme.md) and based on the instructions prepared put the output data files into ```./data/forest/``` folder and run [the notebook](./train_nb-256-32-Forest.ipynb) afterwards.


## Note
If you want to avoid using MATLAB (just using python for all the generating/reconstruction of data) use [this](./Python Version) instead. It's files and functions are edited to work with python totally.

## Resources
- Rocnet notebook https://github.com/ljc91122/RocNet
- https://ljc91122.github.io/publications/rocnet-cviu.pdf

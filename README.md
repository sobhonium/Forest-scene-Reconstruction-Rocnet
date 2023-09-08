# Forest-scene-Reconstruction-Rocnet

The following instrcutions are tested on conda 23.7.2


To run it

```conda env create -n Rocnet-test8 --file ENV.yml```

```pip install -r requirements.txt```

or to do this automatically just run  
```./install.sh```

You can also use Env.yml file and directly:
```conda env create -n <desired-env-name> --file ENV.yml```

Then you can open [the notebook](train_nb-128-32-Forest.ipynb) to run the training.
The notebook is run sample(s) in ```./data/forest/``` folder. To add/replace data you need to convert your sample data into an Octree-based samples. To do so, I recommend to read [this](./data/preprocessing/readme.md) and based on the instructions prepared put the output data files into ```./data/forest/``` folder and run [the notebook](./train_nb-128-32-Forest.ipynb).




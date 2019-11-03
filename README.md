ONSSEN: An Open-source Speech Separation and Enhancement Library
======

Supported Models
------

+ Deep Clustering
+ Chimera Net
+ Chimera++
+ Phase Estimation Network
+ Speech Enhancement with Restoration Layers


Supported Dataset
------

+ Wsj0-2mix (http://www.merl.com/demos/deep-clustering)
+ Daps (https://archive.org/details/daps_dataset)
+ Edinburgh-TTS (https://datashare.is.ed.ac.uk/handle/10283/2791)

Requirements
------
+ PyTorch
+ LibRosa
+ NumPy

Usage
------
You can simply use the existing config JSON file or customize your config file to train the enhancement or separation model.
```
python train.py -c configs/dc_config.json
```


Citing
------

If you use onssen for your research project, please cite one of the following bibtex citations:

    @inproceedings {onssen,
        author = {Zhaoheng Ni and Michael Mandel},
        title = "ONSSEN: An Open-source Speech Separation and Enhancement Library",
        publisher = "under review",
        year = 2019
    }

    @Misc{onssen,
        author = {Zhaoheng Ni and Michael Mandel},
        title = "ONSSEN: An Open-source Speech Separation and Enhancement Library",
        howpublished = {\url{https://github.com/speechLabBcCuny/onssen}},
        year =        {2019}
    }

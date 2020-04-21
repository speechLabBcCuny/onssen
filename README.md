ONSSEN: An Open-source Speech Separation and Enhancement Library
======
Onssen, pronounced as おんせん(温泉, Japanese hot spring), is a PyTorch-based library for speech separation, speech enhancement, or speech style transformation.

Development plan:
------
* [ ] Provide template classes for data, model, and evaluation
* [ ] Move models to separate folders (i.e. Kaldi style)
* [ ] Reproduce scores and upload pretrained models
* [ ] Finish inference method for online separation

2020-04-20 Updates:
-----
+ Add evaluation method for deep clustering
+ Use W_{MR} weight in deep clustering
+ Minor changes


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

    @article{ni2019onssen,
    title={Onssen: an open-source speech separation and enhancement library},
    author={Ni, Zhaoheng and Mandel, Michael I},
    journal={arXiv preprint arXiv:1911.00982},
    year={2019}
    }

    @Misc{onssen,
        author = {Zhaoheng Ni and Michael Mandel},
        title = "ONSSEN: An Open-source Speech Separation and Enhancement Library",
        howpublished = {\url{https://github.com/speechLabBcCuny/onssen}},
        year =        {2019}
    }

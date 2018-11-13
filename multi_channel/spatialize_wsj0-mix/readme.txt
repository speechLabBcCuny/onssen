# Scripts to spatialize the wsj0-mix dataset
#
# Copyright (C) 2017-2018 Mitsubishi Electric Research Labs
#       (Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
###########################################################

These Matlab scripts can be used to build the spatialized wsj0-mix dataset used in:
    Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey, 
    "Multi-Channel Deep Clustering: Discriminative Spectral and Spatial Embeddings for Speaker-Independent Speech Separation," 
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Apr. 2018

###########################################################
Requirements:
    
The scripts require 

- the original wsj0-mix dataset, which can be built using the scripts available at:
    http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip

- Emmanuel Habets's RIR Generator mex code, which can be obtained at:
    https://github.com/ehabets/RIR-Generator
Download rir_generator.cpp under ./RIR-Generator-master/ and compile it in Matlab as follows (if you want to reduce the text output, you can comment out lines 177-181):
>> cd RIR-Generator-master
>> mex rir_generator.cpp
>> cd ..

###########################################################
How to spatialize the wsj0-mix data:

First, modify the following variables in spatialize_wsj0_mix.m according to your settings:
    data_in_root: location of the original wsj0-mix dataset (this should contain subfolders such as 2speakers/ and 3speakers/)
    rir_root: folder in which to save the generated RIRs (it can be the same as data_in_root)
    data_out_root: folder in which to save the spatialized mixtures (it can be the same as data_in_root)

a) If you have the Parallel Processing Toolbox, you can generate the 2-speaker mixtures (num_speakers=2), truncated to the shortest (min_or_max='min'), at 8 kHz (fs=8000) using:

>> spatialize_wsj0_mix(2,'min',8000)

Other types of data (2/3 speakers, 'min'/'max' mixture length, 8/16 kHz sampling rate) can be generated using:

>> spatialize_wsj0_mix(num_speakers,min_or_max,fs)

By default, this uses a pool of 22 workers. The number of workers can be changed in spatialize_wsj0_mix.m by modifying c.NumWorkers at Line 95).

b) If you don't have the Parallel Processing Toolbox, you can use a bash script (launch_spatialize.sh) to launch multiple Matlab jobs in parallel, each of them taking care of a segment of data. 
- Make sure to modify the variables according to the data type you want to generate in launch_spatialize.sh.
- Make sure that the number of workers divides 28000, or uneven splits may occur.

$ ./launch_spatialize.sh

###########################################################
Remarks regarding the data:

The set of sources corresponding to each mixture in the original wsj0-mix data is spatialized using a random setting of the room dimensions, microphone array size and position, speaker position. For each of these settings, we consider two conditions for T60:
- anechoic: the room is assumed anechoic, T60 is set to 0.
- reverb: the room is assumed reverberant, T60 is randomly drawn.
The sampling of the various random parameters is explained in sample_RIRs.m.

Using the rir_info.mat file that we provide here will create 8-channel mixtures for training, development, and evaluation sets. 
Note that:
- constraints on the minimum distance between microphones are only enforced on the first 4 microphones. The last 4 microphones are randomly placed within the sphere determined by the first 2 microphones.
- in our ICASSP 2018 paper, the last 4 microphones were only used for evaluation.
- because we did not use the last 4 microphones in training and development, they were not included when rescaling all waveforms so that the maximum overall was 0.9 to avoid clipping. This may result in a slight difference in scaling for some configurations (i.e., sets of files that correspond to the same original mixture), but all files (mixtures and sources, anechoic and reverberated) are rescaled using the same factor, so this should have no impact on the results.

###########################################################
Expected computation times and sizes for 2 speakers, 'min' length, 8 kHz sampling rate:

Using 22 workers (on a Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz with 24 cores):
- generation of the 28000 RIRs at 8 kHz took about 8 hours
- spatialization of the whole dataset using the generated RIRs took about 22 minutes

Size on disk:
- 22G RIRs_8k/
- 56G 2speakers_anechoic/
- 56G 2speakers_reverb/

To further check that the data generated corresponds to our, we attached two reverberated mixtures and their corresponding RIRs:
- 405o0319_2.3824_01xo030w_-2.3824.wav, spatialized using rir_16000.mat
- 22ho0114_2.3863_22ga0110_-2.3863.wav, spatialized using rir_26000.mat
You may see very small differences on the order of +/-6e-5: if so, it means the original wsj0-mix data you used was generated using a different version of Voicebox from ours. We now provide a specific version of the Voicebox functions used to generate wsj0-mix with our scripts. These differences should not impact the results.

###########################################################
Generating more or different RIRs:

We provide the code (sample_RIRs.m) that was used to randomly generate the information used to create the RIRs, which is stored in the file rir_info.mat that we provide.
Running sample_RIRs as is should exactly reproduce rir_info.mat.
You can also use this code to tweak the settings and generate more RIRs, although be aware that doing so will result in a dataset that is completely different from (and potentially easier or harder than) the one on which we reported our results. Be also aware that some settings may crash the RIR generator (e.g., low T60 in large rooms).



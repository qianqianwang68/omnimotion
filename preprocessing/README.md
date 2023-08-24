# Data processing

This README file contains instructions to compute and process RAFT optical flows for optimizing OmniMotion.

## Data format
The input video data should be organized in the following format:
```
├──sequence_name/
    ├──color/
        ├──00000.jpg
        ├──00001.jpg
        .....
    ├──mask/ (optional; only used for visualization purposes)
        ├──00000.png
        ├──00001.png
        ..... 
```
If you want to run on [DAVIS](https://davischallenge.org/index.html) video sequences, you can run `python get_davis.py <out_dir>` 
which will download the original dataset and organize it in our format for processing. Alternatively, you can 
download some of our processed sequences [here](https://omnimotion.cs.cornell.edu/dataset/) to skip processing and directly start training.

If you want to train on your own video sequence, we recommend you to start with
shorter sequences (< 60 frames) and lower resolution (<= 480p) to manage computational cost. 
You may use `ffmpeg` to extract frames from the video.


## Preparation
The command below moves files to the correct locations and download pretrained models (this only needs to be run once).
```
cd preprocessing/  

mv exhaustive_raft.py filter_raft.py chain_raft.py RAFT/;
cd RAFT; ./download_models.sh; cd ../

mv extract_dino_features.py dino/
```

## Computing and processing flow

Run the following command to process the input video sequence. Please use absolute path for the sequence directory.
```
conda activate omnimotion
python main_processing.py --data_dir <sequence directory> --chain
```
The processing contains several steps:
- computing all pairwise optical flows using `exhaustive_raft.py`
- computing dino features for each frame using `extract_dino_features.py`
- filtering flows using cycle consistency and appearance consistency check using`filter_raft.py`
- (optional) chaining only cycle consistent flows to create denser correspondences using `chain_raft.py`. 
  We found this to be helpful for handling sequences with rapid motion and large displacements. 
  For simple motion, this may be skipped by omitting `--chain` to save processing time. 

After processing the folder should look like the following:
```
├──sequence_name/
    ├──color/
    ├──mask/ (optional; only used for visualization purposes)
    ├──count_maps/
    ├──features/
    ├──raft_exhaustive/
    ├──raft_masks/
    ├──flow_stats.json
```

## Discussion
This processing pipeline is designed to filter and process RAFT optical flow for training our method. 
Our method can also take as input correspondences from other methods, e.g., [TAPIR](https://deepmind-tapir.github.io/) and
[CoTracker](https://co-tracker.github.io/). 
If you want to use different correspondences as input supervision, note that their error patterns might be different from
those of RAFT optical flow, and you may need to devise new filtering methods that are effective for the specific correspondences
you are working with.

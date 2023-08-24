
#Tracking Everything Everywhere All at Once
PyTorch Implementation for paper [Tracking Everything Everywhere All at Once]((https://omnimotion.github.io/)), ICCV 2023.

[Qianqian Wang](https://www.cs.cornell.edu/~qqw/) <sup>1,2</sup>,
[Yen-Yu Chang](https://yuyuchang.github.io/) <sup>1</sup>,
[Ruojin Cai](https://www.cs.cornell.edu/~ruojin/) <sup>1</sup>,
[Zhengqi Li](https://zhengqili.github.io/) <sup>2</sup>,
[Bharath Hariharan](https://www.cs.cornell.edu/~bharathh/) <sup>1</sup>,
[Aleksander Holynski](https://holynski.org/) <sup>2,3</sup>,
[Noah Snavely](https://www.cs.cornell.edu/~snavely/) <sup>1,2</sup>
<br>
<sup>1</sup>Cornell University,  <sup>2</sup>Google Research,  <sup>3</sup>UC Berkeley

#### [Project Page](https://omnimotion.github.io/) | [Paper](https://arxiv.org/pdf/2306.05422.pdf) | [Video](https://www.youtube.com/watch?v=KHoAG3gA024)
## Installation
The code is tested with `python=3.8` and `torch=1.10.0+cu111` on an A100 GPU.
```
git clone --recurse-submodules https://github.com/qianqianwang68/omnimotion/
cd omnimotion/
conda create -n omnimotion python=3.8
conda activate omnimotion
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## Training
1. Please refer to the [preprocessing instructions](preprocessing/README.md) for preparing input data 
   for training OmniMotion. We also provide some processed [data](https://omnimotion.cs.cornell.edu/dataset/)
   that you can download, unzip and directly train on. (Note that depending on the network speed, 
   it may be faster to run the processing script locally than downloading the processed data).
   
2.  With processed input data, run the following command to start training:
    ```
    python train.py --config configs/default.txt --data_dir {sequence_directory}
    ```
    You can view visualizations on tensorboard by running `tensorboard --logdir logs/`. 
    By default, the script trains 100k iterations which takes 8~9h on an A100 GPU.  

## Visualization
The training pipeline generates visualizations (correspondences, pseudo-depth maps, etc) every certain number of steps (saved in `args.out_dir/vis`). 
You can also visualize grid points / trails after training by running: 
```
python viz.py --config configs/default.txt --data_dir {sequence_directory}
```
Make sure `expname` and `data_dir` are correctly specified, so that the
model and data can be loaded. By specifying `expname`, the latest checkpoints that match that `expname` 
will be loaded. Alternatively, you can specify `ckpt_path` to select a particular checkpoint.

To generate the motion trail visualization, foreground/background segmentation mask is required. 
For DAVIS videos one can just use the mask annotations provided by the dataset. For custom videos that don't come with
foreground segmentation masks, you can use [remove.bg](https://www.remove.bg/) to remove the background 
for the query frame, download the masked image and specify its path:
```
python viz.py --config configs/default.txt --data_dir {sequence_directory} --foreground_mask_path {mask_file_path}
```



## Troubleshooting

- The training code utilizes approximately 22GB of CUDA memory. If you encounter CUDA out of memory errors, 
  you may consider reducing the number of sampled points `num_pts` and the chunk size `chunk_size`.
- Due to the highly non-convex nature of the underlying optimization problem, we observe that the optimization process 
  can be sensitive to initialization for certain difficult videos. If you notice significant inaccuracies in surface
  orderings (by examining the pseudo depth maps) persist after 40k steps, 
  it is very likely that training won't recover from that. You may consider restarting the training with a 
  different `loader_seed` to change the initialization. 
  If surfaces are incorrectly put at the nearest depth planes (which are not supposed to be the closest), 
  we found using `mask_near` to disable near samples in the beginning of the training could help in some cases.  
- Another common failure we noticed is that instead of creating a single object in the canonical space with
  correct motion, the method creates duplicated objects in the canonical space with short-ranged motion for each.
  This has to do with both that the input correspondences on the object being sparse and short-ranged, 
  and the optimization being stuck at local minima. This issue may be alleviated with better and longer-range input correspondences 
  such as from [TAPIR](https://deepmind-tapir.github.io/) and [CoTracker](https://co-tracker.github.io/). 
  Alternatively, you may consider adjusting `loader_seed` or the learning rates.


## Citation
```
@article{wang2023omnimotion,
    title   = {Tracking Everything Everywhere All at Once},
    author  = {Wang, Qianqian and Chang, Yen-Yu and Cai, Ruojin and Li, Zhengqi and Hariharan, Bharath and Holynski, Aleksander and Snavely, Noah},
    journal = {ICCV},
    year    = {2023}
}
```




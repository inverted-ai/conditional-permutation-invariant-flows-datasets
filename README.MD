# Conditional sets
This repository provides the code to generate the datasets and validation metrics introduced in ["Conditional Permutation Invariant Flows"](https://openreview.net/forum?id=DUsgPi3oCC) [1]. It contains two main tasks, each consisting of several sub variants. It moreover provides functions to calculate the metrics reported in [1].

The datasets are derivatives of two publicly available datasets. Since both have licenses that do not allow republishing, we provide our code in a way that assumes the user has obtained them from their respective sources. The uses of the two datasets and sub tasks will be described briefly.

If you use this code, please cite our paper:
```
@article{
anonymous2023conditional,
title={Conditional Permutation Invariant Flows},
author={Zwartsenberg, Berend and \'Scibior, Adam and Niedoba, Matthew and Lioutas, Vasileios and Sefas, Justice and Liu, Yunpeng and Dabiri, Setareh and Lavington, Jonathan Wilder and Campbell, Trevor and Wood, Frank},
journal={Submitted to Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=DUsgPi3oCC},
note={Under review}
}
```
## Conditional traffic scenes 
Here, a dataset is provided on top of the INTERACTION dataset. The task is to fit an Nx5 distribution, conditional on the map. Each datapoint is represented as an Nx5 dimensional tensor, with N the number of agents, and the other five dimensions being (x,y,length,width,orientation).
The INTERACTION dataset [2] is required for this task. It's root path is described as `interaction_path`, and should look like:
```
├── interaction_path
│   ├── recorded_trackfiles
│   ├── maps 
│   ├── ...
```
The module `create_traffic_dataset.py` provides a dataset with a `__getitem__` method that returns `location, datapoint`, `location` being the conditional information that has to predict `datatapoint`, an Nx5 dimensional traffic scene. Both `train` and `val` splits are provided.
There are two tasks, the first has variable N, which is not consistent with all generative methods (`prune_outer=0`). The second fixes N=7, which is consistent with more methods (`prune_outer=1`), full details can be found in [1].
The module `infractions.py` provides the metrics reported in the paper.
A convenience script is also provided in `write-datasets.sh` that processes all the datasets and pre-writes the samples.

## Object detection
A dataset based on the CLEVR dataset [3] is provided. This task is to fit an Nx3 distribution conditioned on the input image. N is the number of objects, the other dimensions are [x,y,size].
The CLEVR dataset [3] is required for this task, and should be set through `clevr_path`. The path passed there should look like:
```
├── clevr_path 
│   ├── images
│   ├── scenes 
│   ├── ...
```
The module `create_detection_dataset.py` provides a dataset with a `__getitem__` method, that returns an image path, as well as an Nx3 datapoint. Both `train` and `val` splits are provided through the `split` argument.
Two modes are provided, one where N is fixed to 3 (`task=CLEVR3`), and another where N is 3-6 (`task=CLEVR6`). There is also a function that calculates IOU (intersection over union) reported in the paper.

## Installation
This repository requires `torchdrivesim`. Please follow installation instructions on [https://docs.torchdrivesim.org/en/latest/](https://docs.torchdrivesim.org/en/latest/), and then `pip install -r requirements.txt`.

[1] ["Permutation Invariant Flows"](https://openreview.net/forum?id=DUsgPi3oCC), Zwartsenberg _et al._, TMLR (2023)\
[2] [Interaction Dataset](http://interaction-dataset.com/)\
[3] [CLEVR Dataset](https://cs.stanford.edu/people/jcjohns/clevr/)

# Predicting the virality of TikTok videos

This project has been implemented by Group 31 as a requirement for the Final Project for the Fall 2021 iteration of the CIS-519 course at the University of Pennsylvania.

The reference code for the toolbox LOUPE can be found [here](https://github.com/antoine77340/LOUPE).

The original repository for WILLOW can be found [here](https://github.com/antoine77340/Youtube-8M-WILLOW).

dataset with frame-level feature can be found here https://drive.google.com/drive/folders/1RtwHnztpWYrO32sXMNDFJita8k5Fy545

Video Feature Vector Size for each frame in ViViT: (video_size/patch_size)^2

or Embedding is: (batchSize, numFrames, (video_size/patch_size)^2, 192)

data.npy is of the format (videoId, vid feature, aud feature, label)

## Installation

```
git clone placeholder.git
cd placeholder
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - pytorch
  - pytorch_lightning
  - einops
  - matplotlib
  - numpy
  - scikit-video
  - moviepy
  
</details>

## How To Run?

### Quick Start


---

To train the model: 

```
python train.py
```

---

To test the model: 

```
python test.py
```


### Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv).
```
├── root 
│   ├── dir0
│   ├── dir1
│   ├── dir2
```

### Reproducibility 

Tests that ensure the results of all functions and training loop match the official implentation are contained in a different branch `reproduce`. One can check it out and run the tests:
```
git checkout reproduce
py.test
```

## Method

[Learning To Count Everything](https://github.com/cvlab-stonybrook/LearningToCountEverything)
  

> Basic concept explanation with image


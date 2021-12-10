# Predicting the virality of TikTok videos

This project has been implemented by Group 31 as a requirement for the Final Project for the Fall 2021 iteration of the CIS-519 course at the University of Pennsylvania.

dataset with frame-level feature can be found here https://drive.google.com/drive/folders/1RtwHnztpWYrO32sXMNDFJita8k5Fy545

Video Feature Vector Size for each frame in ViViT: (video_size/patch_size)^2

or Embedding is: (batchSize, numFrames, (video_size/patch_size)^2, 192)

## Installation

```
git clone https://github.com/MeteoRex11/tiktok-virality-prediction
cd tiktok-virality-prediction
pip install einops pytorch_lightning sk-video moviepy
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
#### Model Training & Testing
```
cd ViViT
```
To train the model: 

```
python train.py
```

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

## Method

[ViViT: A Video Vision Transformer](https://arxiv.org/pdf/2103.15691.pdf)
> Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid

![](assets/model.png)

## Citation:
```
@misc{arnab2021vivit,
      title={ViViT: A Video Vision Transformer}, 
      author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
      year={2021},
      eprint={2103.15691},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement:
* Base ViViT code is ported to PyTorch Lightning from [@rishikksh20](https://github.com/rishikksh20) repo : [ViViT-pytorch](https://github.com/rishikksh20/ViViT-pytorch)
* Audio feature extraction taken from Google MediaPipe

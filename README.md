# Predicting the virality of TikTok videos

This project has been implemented by Group 31 as a requirement for the Final Project for the Fall 2021 iteration of the CIS-519 course at the University of Pennsylvania.

## Dataset

The original TikTok dataset with the top 1000 trending videos can be found [here](https://www.kaggle.com/erikvdven/tiktok-trending-december-2020).

The dataset with frame-level features can be found [here](https://drive.google.com/drive/folders/1RtwHnztpWYrO32sXMNDFJita8k5Fy545).

---

## Installation

```
git clone https://github.com/MeteoRex11/tiktok-virality-prediction
cd tiktok-virality-prediction
pip install einops pytorch pytorch_lightning numpy matplotlib sk-video moviepy
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

### 1. Data Pre-Processing

```
run CIS_519_Team_Project_Preprocessing.ipynb
```

---

### 2. Model Training & Testing
```
cd ViViT/
```
To train the model: 

```
python train.py
```

To test the model: 

```
python test.py
```

### 3. Training Log Visualization:
```
tensorboard --logdir lightning_logs
```

---

## Basic Usage:
```python
img = torch.ones([1, 134, 3, 240, 240])

image_size = 240
patch_size = 16
num_classes = 2
num_frames = 134
num_epochs = 30
dim = 128

model = ViViT(image_size, patch_size, num_classes, num_frames, dim)
model = model.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

output = model(img)

print("Shape of model output :", output.shape)      # [B, num_classes]
```
---

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
* Erik van de Ven for scraping the videos for the TikTok dataset

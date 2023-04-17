# Predicting the virality of TikTok videos

This project has been implemented by Group 31 as a requirement for the Final Project for the Fall 2021 iteration of the CIS-519 course at the University of Pennsylvania.

## Dataset

The original TikTok dataset with the top 1000 trending videos can be found [here](https://www.kaggle.com/erikvdven/tiktok-trending-december-2020).

The dataset with frame-level features can be found [here](https://drive.google.com/drive/folders/1RtwHnztpWYrO32sXMNDFJita8k5Fy545).

---

## Installation

```
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - pyroaring
  - sys
  - random
  - matplotlib
  - numpy
  - imageio
  
</details>


### Quick Start



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

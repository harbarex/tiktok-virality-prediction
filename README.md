# Predicting the virality of TikTok videos
This project is being worked on as a requirement for the CIS-519 Final Project by Group 31.

The reference code for the toolbox LOUPE can be found [here](https://github.com/antoine77340/LOUPE).

The original repository for WILLOW can be found [here](https://github.com/antoine77340/Youtube-8M-WILLOW).

dataset with frame-level feature can be found here https://drive.google.com/drive/folders/1RtwHnztpWYrO32sXMNDFJita8k5Fy545

Video Feature Vector Size for each frame in ViViT: (video_size/patch_size)^2

or Embedding is: (batchSize, numFrames, (video_size/patch_size)^2, 192)

data.npy is of the format (videoId, vid feature, aud feature, label)


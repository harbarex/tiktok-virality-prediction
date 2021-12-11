# Video Vision Transformer

Some important notes for the implementation:
* The Embedding Size of the Transformer has a dimension size of (batchSize, numFrames, (video_size/patch_size)^2, dim)
* The training code includes a variable called `include_audio`, which can be set to `True` or `False` to include or exclude Audio features respectively

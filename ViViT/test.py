import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, average_precision_score

from glob import glob
import pickle
from vivit import *
from utils import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Hyperparameters
#................................................................................
image_size = 240
patch_size = 16
num_classes = 2
num_frames = 134
dim = 128

test_batch_number = 49  # Mini-batch number to load

pretrained_path = "lightning_logs/version_18/checkpoints/epoch=29-step=239.ckpt"
#................................................................................


### TESTING PHASE

torch.autograd.set_detect_anomaly(True)

# Define model
model = ViViT(image_size, patch_size, num_classes, num_frames, dim)
model = model.to(device)

# Load Pre-trained Model
checkpoint = torch.load(pretrained_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load test dataset
test_videos, test_labels = load_batch(data_path, 'test', test_batch_number)
test_videos = torch.FloatTensor(test_videos).cuda()
test_videos = test_videos.permute(0, 1, 4, 2, 3)

predictions = model(test_videos)  # [B, num_classes]

y_true = test_labels.astype(int)
y_pred = np.argmax(predictions.cpu().detach().numpy(), 1)
probs = torch.max(predictions, 1).values
probs = probs.cpu().detach().numpy()


"""
Evaluation Metrics:
-------------------

1) Hit@1
2) Precision
3) Recall
4) Global Average Precision
5) Confusion Matrix

"""

hit_at_one = sum(y_true == y_pred)
precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred)
confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

ap = average_precision_score(y_true, probs)

print('\n - Evaluation Metrics -')

print('\nHit@1: {}/{}'.format(hit_at_one, len(y_true)))
print('Precision for Non-Viral Videos: {} & Viral Videos: {}'.format(precision[0], precision[1]))
print('Recall for Non-Viral Videos: {} & Viral Videos: {}'.format(recall[0], recall[1]))

print('Average Precision: ', ap)

print('\nConfusion Matrix for Non-Viral Videos:\n', confusion_matrix[0])
print('\nConfusion Matrix for Viral Videos:\n', confusion_matrix[1])
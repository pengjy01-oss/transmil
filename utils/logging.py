"""CSV metrics logging for training."""
from __future__ import print_function

import csv
import os


def init_metrics_csv(csv_path):
    """Create a new CSV file with the header row."""
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_error',
            'train_acc',
            'train_soft_ratio',
            'train_score_mean',
            'train_topk_mean',
            'train_score_std',
            'val_loss',
            'val_error',
            'val_acc',
            'val_soft_ratio',
            'val_score_mean',
            'val_topk_mean',
            'val_score_std',
            'test_loss',
            'test_error',
            'test_acc',
            'test_soft_ratio',
            'test_score_mean',
            'test_topk_mean',
            'test_score_std',
        ])


def append_metrics_csv(csv_path, epoch, train_loss=None, train_error=None, train_acc=None,
                       train_soft_ratio=None, train_score_mean=None, train_topk_mean=None, train_score_std=None,
                       val_loss=None, val_error=None, val_acc=None,
                       val_soft_ratio=None, val_score_mean=None, val_topk_mean=None, val_score_std=None,
                       test_loss=None, test_error=None, test_acc=None,
                       test_soft_ratio=None, test_score_mean=None, test_topk_mean=None, test_score_std=None):
    """Append one row of epoch metrics to the CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            train_error,
            train_acc,
            train_soft_ratio,
            train_score_mean,
            train_topk_mean,
            train_score_std,
            val_loss,
            val_error,
            val_acc,
            val_soft_ratio,
            val_score_mean,
            val_topk_mean,
            val_score_std,
            test_loss,
            test_error,
            test_acc,
            test_soft_ratio,
            test_score_mean,
            test_topk_mean,
            test_score_std,
        ])

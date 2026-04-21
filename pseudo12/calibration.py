"""Pre-training calibration pass to collect severity scores and generate pseudo-12 labels."""

import os
import numpy as np
import torch
from torch.autograd import Variable

from pseudo12.pseudo_labels import (
    compute_severity_score,
    calibrate_thresholds,
    generate_pseudo12_labels,
    print_pseudo12_distribution,
    save_thresholds,
    NUM_SUBTYPES,
)


def calibration_pass(model, data_loader, args, score_weights=None):
    """Run one forward pass over data_loader to collect burden/coverage stats.

    Returns:
        severity_scores: list of float (one per sample)
        main_labels: list of int (one per sample)
    """
    model.eval()
    severity_scores = []
    main_labels = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            data = batch_data[0]
            label = batch_data[1]
            pos_z = batch_data[2] if len(batch_data) > 2 else None
            bag_label = label[0] if isinstance(label, (list, tuple)) else label

            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
                if pos_z is not None:
                    pos_z = pos_z.cuda()

            _ = model(Variable(data), pos_z=pos_z)
            aux = model.get_latest_aux_outputs()
            burden_stats = aux.get('burden_stats', None)
            coverage_stats = aux.get('coverage_stats', None)

            S = compute_severity_score(burden_stats, coverage_stats, weights=score_weights)
            severity_scores.append(S)
            main_labels.append(int(bag_label.view(-1).cpu().item()))

    return severity_scores, main_labels


def build_pseudo12_labels_for_all_splits(
    model, train_loader, val_loader, test_loader, args,
    score_weights=None,
    quantiles=(1.0 / 3.0, 2.0 / 3.0),
    save_path=None,
):
    """Generate pseudo-12 labels for train/val/test using training-set thresholds only.

    Returns:
        train_pseudo12: np.ndarray of int64
        val_pseudo12: np.ndarray of int64 or None
        test_pseudo12: np.ndarray of int64
        thresholds: dict {cls: (t1, t2)}
    """
    print('\n====== Pseudo-12 Calibration Pass ======')

    # Step 1: Collect severity scores from training set
    print('Running calibration on training set ({} samples)...'.format(len(train_loader)))
    train_scores, train_labels = calibration_pass(model, train_loader, args, score_weights)

    # Step 2: Compute thresholds from training set ONLY
    scores_by_class = {}
    for s, y in zip(train_scores, train_labels):
        scores_by_class.setdefault(y, []).append(s)

    print('\nSeverity score statistics per class (training set):')
    for cls in range(4):
        arr = np.array(scores_by_class.get(cls, []))
        if len(arr) > 0:
            print('  Stage {}: n={}, mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
                cls, len(arr), arr.mean(), arr.std(), arr.min(), arr.max()
            ))

    thresholds = calibrate_thresholds(scores_by_class, quantiles=quantiles)
    print('\nPseudo-12 thresholds (from training set):')
    for cls in range(4):
        t1, t2 = thresholds[cls]
        print('  Stage {}: t1={:.4f}, t2={:.4f}'.format(cls, t1, t2))

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_thresholds(thresholds, save_path)

    # Step 3: Assign pseudo-12 labels using training-set thresholds
    train_pseudo12 = generate_pseudo12_labels(train_scores, train_labels, thresholds)
    print_pseudo12_distribution(train_pseudo12, train_labels, split_name='Train')

    # Step 4: Val set (use training thresholds)
    val_pseudo12 = None
    if val_loader is not None:
        print('Running calibration on validation set ({} samples)...'.format(len(val_loader)))
        val_scores, val_labels = calibration_pass(model, val_loader, args, score_weights)
        val_pseudo12 = generate_pseudo12_labels(val_scores, val_labels, thresholds)
        print_pseudo12_distribution(val_pseudo12, val_labels, split_name='Val')

    # Step 5: Test set (use training thresholds)
    print('Running calibration on test set ({} samples)...'.format(len(test_loader)))
    test_scores, test_labels = calibration_pass(model, test_loader, args, score_weights)
    test_pseudo12 = generate_pseudo12_labels(test_scores, test_labels, thresholds)
    print_pseudo12_distribution(test_pseudo12, test_labels, split_name='Test')

    print('====== Pseudo-12 Calibration Done ======\n')
    return train_pseudo12, val_pseudo12, test_pseudo12, thresholds

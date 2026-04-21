"""Pre-training calibration pass for soft 12-type target generation (Plan C).

Reuses the forward-pass logic from pseudo12.calibration but produces
continuous soft target distributions instead of hard integer labels.
"""

import os
import numpy as np
import torch
from torch.autograd import Variable

from soft12.soft_labels import (
    compute_severity_score,
    compute_normalization_stats,
    generate_soft12_targets,
    print_soft12_diagnostics,
    save_norm_stats,
    NUM_MAIN_CLASSES,
)


def calibration_pass(model, data_loader, args, score_weights=None):
    """Run one forward pass over data_loader to collect burden/coverage stats.

    Identical to pseudo12.calibration.calibration_pass — reused here for
    self-containment.

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

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(data_loader):
                print('  calibration: {}/{}'.format(batch_idx + 1, len(data_loader)), flush=True)

    return severity_scores, main_labels


def build_soft12_targets_for_all_splits(
    model,
    train_loader,
    val_loader,
    test_loader,
    args,
    score_weights=None,
    tau=0.5,
    eps=1e-6,
    save_path=None,
    n_examples=5,
):
    """Generate soft 12-type target distributions for train/val/test.

    Plan C calibration:
    1. Forward pass through training set → collect S scores.
    2. Compute S_min / S_max per class from TRAINING SET ONLY.
    3. For each sample in all splits: normalize S → r, then compute Gaussian
       soft distribution → embed into 12-dim target.

    Returns:
        train_soft12: np.ndarray [N_train, 12], float32
        val_soft12:   np.ndarray [N_val, 12] or None
        test_soft12:  np.ndarray [N_test, 12], float32
        norm_stats:   dict {cls: (S_min, S_max)}
    """
    print('\n====== Soft-12 Calibration Pass (Plan C) ======')
    print('Score weights: {}'.format(score_weights))
    print('tau = {:.4f}, eps = {:.2e}'.format(tau, eps))

    # Step 1: Collect severity scores from training set
    print('Running calibration on training set ({} samples)...'.format(len(train_loader)))
    train_scores, train_labels = calibration_pass(model, train_loader, args, score_weights)

    # Step 2: Compute normalization stats from TRAINING SET ONLY
    scores_by_class = {}
    for s, y in zip(train_scores, train_labels):
        scores_by_class.setdefault(y, []).append(s)

    print('\nSeverity score statistics per class (training set):')
    for cls in range(NUM_MAIN_CLASSES):
        arr = np.array(scores_by_class.get(cls, []))
        if len(arr) > 0:
            print('  Stage {}: n={}, mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
                cls, len(arr), arr.mean(), arr.std(), arr.min(), arr.max()
            ))

    norm_stats = compute_normalization_stats(scores_by_class, eps=eps)
    print('\nSoft-12 normalization stats (S_min, S_max) per class:')
    for cls in range(NUM_MAIN_CLASSES):
        s_min, s_max = norm_stats[cls]
        print('  Stage {}: S_min={:.4f}, S_max={:.4f}'.format(cls, s_min, s_max))

    if save_path:
        dir_part = os.path.dirname(save_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        save_norm_stats(norm_stats, save_path)

    # Step 3: Generate soft targets for training set
    train_soft12 = generate_soft12_targets(train_scores, train_labels, norm_stats, tau=tau, eps=eps)
    print_soft12_diagnostics(
        train_soft12, train_scores, train_labels, norm_stats,
        tau=tau, split_name='Train', n_examples=n_examples
    )

    # Step 4: Val set (use training norm_stats)
    val_soft12 = None
    if val_loader is not None:
        print('Running calibration on validation set ({} samples)...'.format(len(val_loader)))
        val_scores, val_labels = calibration_pass(model, val_loader, args, score_weights)
        val_soft12 = generate_soft12_targets(val_scores, val_labels, norm_stats, tau=tau, eps=eps)
        print_soft12_diagnostics(
            val_soft12, val_scores, val_labels, norm_stats,
            tau=tau, split_name='Val', n_examples=n_examples
        )

    # Step 5: Test set (use training norm_stats)
    print('Running calibration on test set ({} samples)...'.format(len(test_loader)))
    test_scores, test_labels = calibration_pass(model, test_loader, args, score_weights)
    test_soft12 = generate_soft12_targets(test_scores, test_labels, norm_stats, tau=tau, eps=eps)
    print_soft12_diagnostics(
        test_soft12, test_scores, test_labels, norm_stats,
        tau=tau, split_name='Test', n_examples=n_examples
    )

    print('====== Soft-12 Calibration Done ======\n')
    return train_soft12, val_soft12, test_soft12, norm_stats

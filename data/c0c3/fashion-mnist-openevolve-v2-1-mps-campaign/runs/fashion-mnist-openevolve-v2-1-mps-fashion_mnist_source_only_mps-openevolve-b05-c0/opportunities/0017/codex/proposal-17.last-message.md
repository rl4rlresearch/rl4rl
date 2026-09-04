MECHANISM: Moderate decoupled weight regularization

HYPOTHESIS: Increasing AdamW weight decay to 0.01 will exceed 9,254 correct predictions by improving generalization without weakening targets or altering the proven training path.

INTENDED_EDIT: Change only AdamW weight decay from 0.0001 to 0.01.

EVIDENCE: Label smoothing 0.05 outperformed both hard targets and 0.10 smoothing, showing that moderate regularization benefits accuracy; the current 0.0001 weight decay is negligible over this training budget, motivating a modest orthogonal increase.

<<<<<<< SEARCH
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-2)
>>>>>>> REPLACE
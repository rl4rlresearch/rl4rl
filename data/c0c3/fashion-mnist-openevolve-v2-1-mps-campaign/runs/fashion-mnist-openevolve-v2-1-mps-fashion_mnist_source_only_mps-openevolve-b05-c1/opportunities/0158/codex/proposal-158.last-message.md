MECHANISM: Longer-horizon late-training parameter EMA

HYPOTHESIS: Increasing the EMA decay cap from 0.98 to 0.99 will exceed 9,360 correct predictions by smoothing more of the low-learning-rate final trajectory while preserving the verified architecture and unaveraged BatchNorm statistics.

INTENDED_EDIT: Change only the parameter EMA decay cap from 0.98 to 0.99.

EVIDENCE: The verified 0.98 design achieved 9,360 correct; architecture, loss, probability-TTA, and BatchNorm-buffer changes all performed worse. The sole 0.99 attempt timed out and therefore provides no contrary performance evidence.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE
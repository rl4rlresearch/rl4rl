MECHANISM: Successive late-boundary causal step pruning

HYPOTHESIS: Removing the new final scheduled frame (index 27) will preserve at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and cutting approximately 12 million dense inference MACs.

INTENDED_EDIT: End the contiguous schedule one frame earlier, retaining frames 3–26 and the proven 32/32/31 GRU architecture.

EVIDENCE: Removing late frame 28 retained 85.28% accuracy, whereas removing early frame 3 fell to 84.79%; this suggests another late-boundary deletion is the most informative one-step reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames - 4))
=======
        return list(range(3, available_frames - 5))
>>>>>>> REPLACE
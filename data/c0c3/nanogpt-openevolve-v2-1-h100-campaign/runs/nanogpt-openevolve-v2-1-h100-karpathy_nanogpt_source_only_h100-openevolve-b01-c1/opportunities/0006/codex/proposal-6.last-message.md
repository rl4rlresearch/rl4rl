MECHANISM: Stacked local-attention receptive field

HYPOTHESIS: Making the final layer local will increase throughput beyond 530.6M tokens while eight stacked 256-token windows retain sequence-wide effective reach, lowering val_bpb below 0.992286.

INTENDED_EDIT: Use 256-token local attention in all eight layers by removing the final full-context override.

EVIDENCE: Replacing one of two full-context layers with a 256-token local layer increased tokens from 522.2M to 530.6M and improved val_bpb from 0.993215 to 0.992286, motivating a direct test of whether the remaining global layer is necessary.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
        return window_sizes
>>>>>>> REPLACE
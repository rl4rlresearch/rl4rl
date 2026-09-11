MECHANISM: Fully local stacked receptive-field attention

HYPOTHESIS: Making all eight layers use 512-token attention will exceed 518.5M training tokens, while the stacked receptive field still spans the full 2,048-token sequence, lowering val_bpb below 0.993870.

INTENDED_EDIT: Remove the forced full-context final layer so the existing SSSS pattern produces eight quarter-context attention layers.

EVIDENCE: Replacing the intermediate full-context layer with a 512-token local layer improved val_bpb from 0.994770 to 0.993870 and throughput from 503.3M to 518.5M tokens; seven stacked local layers already provide a full-sequence receptive field before the current final anchor.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven quarter-context layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # eight quarter-context layers; stacked receptive field spans the sequence
>>>>>>> REPLACE
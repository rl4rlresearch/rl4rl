MECHANISM: Broader sliding-window attention

HYPOTHESIS: Doubling short-layer context from 512 to 1024 tokens will retain at least 500M training tokens and reduce val_bpb below 0.991682.

INTENDED_EDIT: Keep the best 2.25×/2.75×/5.5× architecture and SSSL pattern, but make each short-attention layer cover half rather than one quarter of the 2048-token sequence.

EVIDENCE: Removing the layer-3 full-context pass increased throughput only from 510.7M to 513.8M tokens but worsened val_bpb from 0.991682 to 0.992320, indicating that additional contextual mixing is worth modest attention cost.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 2
>>>>>>> REPLACE
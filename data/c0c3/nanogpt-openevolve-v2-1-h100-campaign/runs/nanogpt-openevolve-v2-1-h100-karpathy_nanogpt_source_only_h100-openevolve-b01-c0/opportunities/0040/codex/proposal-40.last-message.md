MECHANISM: Finer layer-6 context–throughput interpolation

HYPOTHESIS: A 768-token window at layer 6 will beat 0.984059 val_bpb by retaining useful mid-range context while recovering throughput toward the 506.2M-token 512-window design.

INTENDED_EDIT: Reduce the layer-6 medium attention window from 1024 to 768 tokens; layers 4 and 8 remain full-attention and all other layers remain at 512 tokens.

EVIDENCE: Increasing layer 6 from 512 to 1024 tokens improved val_bpb from 0.984313 to 0.984059 while reducing throughput from 506.2M to 500.2M tokens, whereas increasing it further to 2048 regressed to 0.984244; 768 tests the unmeasured interval on the improving side of the observed optimum.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = 3 * long_window // 8
>>>>>>> REPLACE
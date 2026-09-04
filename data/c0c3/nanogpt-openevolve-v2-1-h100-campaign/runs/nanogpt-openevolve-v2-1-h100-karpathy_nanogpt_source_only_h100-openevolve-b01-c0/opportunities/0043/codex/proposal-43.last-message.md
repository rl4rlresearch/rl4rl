MECHANISM: Fine-grained upper-side context interpolation

HYPOTHESIS: A 1088-token layer-6 window will beat 0.984059 val_bpb by adding modest context beyond 1024 while avoiding the throughput loss observed at 1280 tokens.

INTENDED_EDIT: Increase only the medium attention window used by layer 6 from 1024 to 1088 tokens; layers 4 and 8 remain full-attention and all other layers remain at 512 tokens.

EVIDENCE: The 1024-token window achieved 0.984059, while 960 was narrowly worse at 0.984117 and 1280 regressed to 0.984663; testing 1088 is the closest unmeasured upper neighbor and refines the observed optimum.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = long_window * 17 // 32
>>>>>>> REPLACE
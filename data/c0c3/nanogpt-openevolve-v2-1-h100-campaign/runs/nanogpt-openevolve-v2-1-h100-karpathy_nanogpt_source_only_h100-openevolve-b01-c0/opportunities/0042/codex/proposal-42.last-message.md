MECHANISM: Fine-grained layer-6 context–throughput interpolation

HYPOTHESIS: A 960-token layer-6 window will beat 0.984059 val_bpb by preserving nearly all useful context from 1024 tokens while modestly increasing training throughput.

INTENDED_EDIT: Reduce only layer 6’s medium attention window from 1024 to 960 tokens; layers 4 and 8 remain full-attention and all other layers remain at 512 tokens.

EVIDENCE: The 1024-token window is best; equally spaced tests at 768 and 1280 regressed to 0.984493 and 0.984663 respectively, with the smaller-window side performing better. Testing 960 probes just below the observed optimum at finer resolution.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = long_window * 15 // 32
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 960-token attention at layer 6
>>>>>>> REPLACE
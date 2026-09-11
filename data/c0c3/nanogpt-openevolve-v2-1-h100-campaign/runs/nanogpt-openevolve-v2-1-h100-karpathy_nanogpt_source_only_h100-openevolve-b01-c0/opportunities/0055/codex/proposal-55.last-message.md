MECHANISM: Fine-grained post-global context interpolation

HYPOTHESIS: A 320-token layer-5 attention window will beat 0.983955 val_bpb by recovering some context from the narrowly inferior 384-token design while retaining most of the throughput benefit of 256-token attention.

INTENDED_EDIT: Increase only layer 5’s attention window from 256 to 320 tokens; retain full attention at layers 4 and 8, 1024-token attention at layer 6, and 512-token attention elsewhere.

EVIDENCE: Layer 5 performed best at 256 tokens with 0.983955 val_bpb; 384 tokens was only 0.000091 worse, whereas 128 tokens regressed by 0.000342. This asymmetric neighborhood motivates testing the aligned 320-token midpoint on the higher-context side.

<<<<<<< SEARCH
        narrow_window = long_window // 8
        short_window = long_window // 4
=======
        narrow_window = 5 * long_window // 32
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLNMSL" # 256 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
=======
WINDOW_PATTERN = "SSSLNMSL" # 320 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
>>>>>>> REPLACE
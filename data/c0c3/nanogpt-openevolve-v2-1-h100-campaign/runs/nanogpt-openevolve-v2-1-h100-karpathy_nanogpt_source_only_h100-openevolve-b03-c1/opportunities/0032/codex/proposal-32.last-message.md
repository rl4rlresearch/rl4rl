MECHANISM: Distributed medium-context attention

HYPOTHESIS: Widening each short-context layer from 512 to 768 tokens while retaining the proven global layers at indices 3, 5, and 7 will reduce val_bpb below 0.984227 by distributing additional context across five layers.

INTENDED_EDIT: Change short attention windows from one-quarter to three-eighths of the 2048-token sequence, preserving all other architecture and training settings.

EVIDENCE: Adding a third global layer improved val_bpb from 0.984312 to 0.984227 despite reducing processed tokens from 501.6M to 495.5M, showing that extra attention context can outweigh its throughput cost; widening local windows tests a more distributed allocation of comparable attention compute.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
=======
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=three-eighths context
>>>>>>> REPLACE
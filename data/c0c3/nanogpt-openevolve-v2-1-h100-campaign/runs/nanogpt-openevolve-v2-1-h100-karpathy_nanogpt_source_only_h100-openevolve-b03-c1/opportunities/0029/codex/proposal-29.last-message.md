MECHANISM: Denser late global-context mixing

HYPOTHESIS: Adding one well-spaced full-context attention layer will reduce val_bpb below 0.984312 because the measured penalty from removing global mixing outweighed its throughput gain.

INTENDED_EDIT: Preserve the architecture, optimizer, batch size, and 78% linear warmdown while changing the eight-layer attention topology from two to three full-context layers at indices 3, 5, and 7.

EVIDENCE: Removing global mixing worsened val_bpb to 0.994122 despite processing more tokens, indicating that additional long-range information flow may be worth a modest throughput cost.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
>>>>>>> REPLACE
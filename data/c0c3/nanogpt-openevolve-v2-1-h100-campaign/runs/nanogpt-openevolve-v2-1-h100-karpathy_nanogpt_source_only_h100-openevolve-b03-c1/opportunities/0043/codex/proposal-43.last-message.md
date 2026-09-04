MECHANISM: Alternating late-stage medium/global context refinement

HYPOTHESIS: Giving layer 4 a 1024-token window will reduce val_bpb below 0.984083 by adding targeted post-global refinement without broadly widening all local layers.

INTENDED_EDIT: Change layer 4 from a 512-token short window to a 1024-token medium window, preserving full-context layers 3, 5, and 7 and the proven medium window at layer 6.

EVIDENCE: A single 1024-token window at layer 6 improved val_bpb from 0.984227 to 0.984083, whereas widening every short layer to 768 regressed to 0.985663; this motivates another concentrated medium window rather than distributed widening.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLML" # full context at 3,5,7; half context at 6
=======
WINDOW_PATTERN = "SSSLMLML" # full context at 3,5,7; half context at 4,6
>>>>>>> REPLACE
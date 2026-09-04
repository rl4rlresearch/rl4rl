MECHANISM: Asymmetric pre-consolidation widening

HYPOTHESIS: Retaining the 512-token window before layer 4 while reverting layer 7 to 256 tokens will lower val_bpb below 0.983720 because early enriched features can propagate through the second stage, while the late wider window is partly redundant with layer 8’s full-context attention.

INTENDED_EDIT: Change the eight-layer window schedule from SSMLSSML to SSMLSSSL, preserving the first medium window and recovering throughput from the second.

EVIDENCE: SSML improved val_bpb from 0.984467 to 0.983720 while processing fewer tokens; isolating its earlier widening tests whether that benefit can be retained with throughput closer to the 256-token baseline.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSML" # widen local context immediately before global layers 4 and 8
=======
WINDOW_PATTERN = "SSMLSSSL" # widen context only before the first global consolidation
>>>>>>> REPLACE
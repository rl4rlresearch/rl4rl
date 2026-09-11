MECHANISM: Pre-consolidation context widening

HYPOTHESIS: Using 512-token attention only in the local layers immediately preceding full-context layers 4 and 8 will lower val_bpb below 0.984467 by enriching inputs to global consolidation while retaining more throughput than uniformly widening all six local layers.

INTENDED_EDIT: Add a medium 512-token window type and change the pattern from SSSL to SSML, yielding 256, 256, 512, 2048-token windows in each four-layer stage.

EVIDENCE: Uniform 512-token local windows nearly matched the best result at 0.984862 but reduced total tokens to 503.6M, while 256-token windows with full-context layers 4 and 8 achieved 0.984467 at 516.9M tokens; concentrating wider attention immediately before those successful global layers tests whether its contextual benefit can be retained at lower cost.

<<<<<<< SEARCH
    window_pattern: str = "SSSL"
=======
    window_pattern: str = "SSML"
>>>>>>> REPLACE

<<<<<<< SEARCH
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 4
        short_window = long_window // 8
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
=======
WINDOW_PATTERN = "SSML" # widen local context immediately before global layers 4 and 8
>>>>>>> REPLACE
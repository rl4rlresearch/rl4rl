MECHANISM: Penultimate global integration with local refinement

HYPOTHESIS: Moving the sole full-context attention layer from the final block to the penultimate block will preserve global information while giving the final local block an opportunity to refine globally integrated representations, lowering val_bpb below 0.992286 without changing attention compute.

INTENDED_EDIT: Keep seven 256-token local layers and one 2048-token global layer, but place the global layer at depth 7 and make the final layer local.

EVIDENCE: Removing the sole global layer regressed val_bpb from 0.992286 to 1.015479 despite higher throughput, proving that global integration is essential; moving rather than removing it isolates whether its final-layer placement is optimal at equal compute and parameter count.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
        # Integrate globally in the penultimate block, then locally refine those
        # globally contextualized representations in the final block.
        window_sizes[-2] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "S"    # all local attention; final layer is forced to full context
=======
WINDOW_PATTERN = "S"    # all local attention; penultimate layer is forced to full context
>>>>>>> REPLACE
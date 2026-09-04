MECHANISM: Coverage-preserving late-cluster thinning

HYPOTHESIS: Removing frame 23 from the passing 22-step schedule will retain at least 85% validation accuracy while reducing execution to 21 steps and total inference MACs by approximately 4.5%, from 288.97M to 275.93M.

INTENDED_EDIT: Preserve frames 0, 4, 27, 28, and 31 while pruning frame 23 from the densely sampled 21–24 region, leaving the architecture and training procedure unchanged.

EVIDENCE: The current 22-step model achieved 86.63% accuracy after late-frame pruning, leaving 1.63 percentage points of margin; pruning an interior frame from a four-frame consecutive cluster tests further temporal redundancy without enlarging the successful onset gap or final 28-to-31 gap.

<<<<<<< SEARCH
        if len(indices) > 25:
            del indices[1:4]
            del indices[-2]
        return indices
=======
        if len(indices) > 25:
            del indices[1:4]
            del indices[-7]
            del indices[-2]
        return indices
>>>>>>> REPLACE
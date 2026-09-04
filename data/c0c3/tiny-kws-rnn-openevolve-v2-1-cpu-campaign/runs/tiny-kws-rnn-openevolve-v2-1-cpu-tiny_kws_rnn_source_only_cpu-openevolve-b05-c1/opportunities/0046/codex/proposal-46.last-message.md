MECHANISM: Endpoint-preserving late-frame pruning

HYPOTHESIS: Removing frame 29 from the passing 23-step schedule will retain at least 85% validation accuracy while reducing execution to 22 steps and total inference MACs by approximately 4.3%, from 302.01M to 288.97M.

INTENDED_EDIT: Preserve the verified schedule, including frames 0, 4, and 31, while deleting one densely sampled late frame immediately before the final endpoint.

EVIDENCE: The 80-unit signed-extrema model achieved 86.38% accuracy at 23 steps; prior 24-step failure followed loss of early coverage, whereas coverage-preserving resampling passed, motivating pruning a redundant late frame without disturbing the successful onset pattern.

<<<<<<< SEARCH
        if len(indices) > 25:
            del indices[1:4]
        return indices
=======
        if len(indices) > 25:
            del indices[1:4]
            del indices[-2]
        return indices
>>>>>>> REPLACE
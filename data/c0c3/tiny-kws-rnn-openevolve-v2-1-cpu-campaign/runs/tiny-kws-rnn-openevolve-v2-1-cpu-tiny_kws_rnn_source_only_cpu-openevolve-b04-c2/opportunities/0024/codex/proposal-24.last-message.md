MECHANISM: Incremental trailing-frame trim at the qualified frontier

HYPOTHESIS: The 103-unit GRU trained at batch size 64 on frames 4–26 will retain at least 85% validation accuracy while reducing inference to 23 recurrent steps and approximately 713.1 million total MACs.

INTENDED_EDIT: Schedule 23 contiguous frames for standard recordings by omitting frames 27–31, preserving the architecture and training procedure.

EVIDENCE: The qualified frames 4–27 design achieved 86.26% accuracy at 24 steps, leaving 1.26 percentage points of margin; removing the adjacent trailing frame is the smallest untested structural reduction below that frontier.

<<<<<<< SEARCH
        return list(range(4, available_frames - 2))
=======
        return list(range(4, available_frames - 5))
>>>>>>> REPLACE
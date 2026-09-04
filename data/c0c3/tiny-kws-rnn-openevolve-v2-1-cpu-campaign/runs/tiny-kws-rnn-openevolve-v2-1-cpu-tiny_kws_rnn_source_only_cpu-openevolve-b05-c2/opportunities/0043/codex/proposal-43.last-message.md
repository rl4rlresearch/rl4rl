MECHANISM: Dense-cluster temporal pruning

HYPOTHESIS: Omitting frame 6 instead of accuracy-critical frame 29 will retain at least 85% validation accuracy while executing 21 recurrent steps and approximately 277,699,840 total inference MACs.

INTENDED_EDIT: Extend the qualified 22-step schedule by omitting frame 6, which is surrounded by retained frames, while preserving frame 29 and both causal endpoints.

EVIDENCE: The qualified 22-step design achieved 86.01% accuracy; its 21-step variant reached only 84.91% after omitting frame 29, motivating relocation of the same compute reduction to densely sampled frame 6.

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE
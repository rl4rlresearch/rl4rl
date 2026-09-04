MECHANISM: Dense-cluster temporal pruning

HYPOTHESIS: Omitting frame 6 instead of the accuracy-critical frame 29 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 277,699,840.

INTENDED_EDIT: Add a three-sixteenths omission to the qualified 22-step schedule, removing a frame surrounded by retained neighbors while preserving frame 29 and both causal endpoints.

EVIDENCE: The 22-step schedule achieved 86.01% accuracy, whereas its 21-step variant fell to 84.91% after removing frame 29; this suggests testing the same compute reduction at a densely sampled interior location.

<<<<<<< SEARCH
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames >= 32:
                skipped.add(available_frames // 32)
                skipped.add(3 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE
MECHANISM: Symmetric boundary-frame temporal pruning

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 21 causal frames while reducing total inference MACs from 290,844,160 to approximately 277,699,840.

INTENDED_EDIT: Omit frame 29, the latest remaining near-boundary interior frame, while preserving both causal endpoints and reducing execution from 22 to 21 recurrent steps.

EVIDENCE: The 22-step design achieved 86.01% accuracy after every adjacent reduction from 32 through 22 steps qualified; complementing the omitted early frames 1 and 2 with late frame 29 is the smallest next structural reduction.

<<<<<<< SEARCH
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames >= 32:
                skipped.add(available_frames // 32)
                skipped.add(29 * available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE
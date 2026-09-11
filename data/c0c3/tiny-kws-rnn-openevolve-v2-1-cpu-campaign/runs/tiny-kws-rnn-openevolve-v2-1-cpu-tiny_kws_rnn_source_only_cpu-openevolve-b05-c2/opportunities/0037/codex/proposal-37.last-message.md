MECHANISM: Uniform one-in-four temporal pruning with signed-extrema readout

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 25 causal frames while reducing total inference MACs from 343,421,440 to approximately 330,277,120.

INTENDED_EDIT: Add the missing five-eighths-frame omission, producing evenly spaced omissions at every fourth interior frame while retaining both endpoints.

EVIDENCE: The 64-unit design qualified at every incremental reduction from 32 through 26 steps; the current 26-step result achieved 85.77% accuracy, so the adjacent 25-step schedule is the most informative next temporal-compute boundary.

<<<<<<< SEARCH
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
=======
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
>>>>>>> REPLACE
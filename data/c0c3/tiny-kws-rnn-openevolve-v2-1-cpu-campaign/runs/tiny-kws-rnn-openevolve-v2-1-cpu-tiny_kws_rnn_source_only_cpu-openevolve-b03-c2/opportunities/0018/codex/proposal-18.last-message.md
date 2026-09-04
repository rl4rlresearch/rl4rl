MECHANISM: Nested trailing-frame causal subsampling

HYPOTHESIS: Removing frame 28 from the current 32-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.85%.

INTENDED_EDIT: Extend the qualified nested schedule to 25 steps by additionally omitting `available_frames - 4`, preserving the 110-unit dual-view GRU and training procedure.

EVIDENCE: The current 26-step design achieved 86.63% accuracy; the preceding isolated 27-to-26-step reduction lost 1.35 points, leaving enough observed margin to test one further trailing-edge omission.

<<<<<<< SEARCH
                not in (1, available_frames - 3, available_frames - 2)
=======
                not in (
                    1,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
>>>>>>> REPLACE
MECHANISM: Nested leading-edge frame subsampling

HYPOTHESIS: Omitting frame 3 from the qualified 110-unit, 23-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 4.35%.

INTENDED_EDIT: Use 22 recurrent steps by excluding frames 1–3 and 27–30 for 32-frame inputs, while preserving the qualified dual-readout GRU and training procedure.

EVIDENCE: The qualified 23-step design achieved 85.52% accuracy after removing frame 2 from the 24-step schedule, slightly improving accuracy; this supports testing one further adjacent early-frame omission while retaining the more informative late frames implicated by the failed frame-25 removal.

<<<<<<< SEARCH
                not in (
                    1,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
=======
                not in (
                    1,
                    2,
                    3,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
>>>>>>> REPLACE
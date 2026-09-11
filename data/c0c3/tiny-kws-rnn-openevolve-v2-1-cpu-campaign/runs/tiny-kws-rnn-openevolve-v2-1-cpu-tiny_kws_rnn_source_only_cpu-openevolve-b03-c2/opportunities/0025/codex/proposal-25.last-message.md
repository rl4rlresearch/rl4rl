MECHANISM: Leading-silence frame omission

HYPOTHESIS: Omitting frame 0 from the qualified 110-unit, 23-step schedule while retaining frame 3 will achieve at least 85% validation accuracy with 22 recurrent steps and approximately 4.35% fewer recurrent MACs.

INTENDED_EDIT: Exclude frame 0 in addition to frames 1, 2, and 27–30, preserving the qualified dual-readout GRU and training procedure.

EVIDENCE: The 23-step model qualified at 85.52% after frames 1 and 2 were removed, whereas removing frame 3 reduced accuracy to 84.54%; this isolates whether the remaining earliest frame is less informative than frame 3 at the same 22-step cost.

<<<<<<< SEARCH
                not in (
                    1,
                    2,
                    available_frames - 5,
=======
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
>>>>>>> REPLACE
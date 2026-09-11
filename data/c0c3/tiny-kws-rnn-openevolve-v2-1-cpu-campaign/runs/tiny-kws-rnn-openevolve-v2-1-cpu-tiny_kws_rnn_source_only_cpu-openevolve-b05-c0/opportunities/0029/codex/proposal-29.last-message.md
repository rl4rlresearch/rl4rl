MECHANISM: Redundant mid-tail frame removal

HYPOTHESIS: Processing frames 2–23, 25, and 29 will achieve at least 85% validation accuracy with 24 recurrent steps by preserving the successful dense prefix and temporal endpoint while omitting the likely redundant intermediate tail frame 27.

INTENDED_EDIT: Change the 24-frame schedule to retain frame 23 and remove frame 27 from the sparse tail.

EVIDENCE: The verified 25-step schedule using frames 2–23, 25, 27, and 29 achieved 85.52%, while removing frame 23 narrowly failed at 84.91%; this suggests preserving dense-prefix continuity and testing redundancy among the late samples.

<<<<<<< SEARCH
            return full_window[:-6] + full_window[-5::2]
=======
            return full_window[:-6] + [full_window[-5], full_window[-1]]
>>>>>>> REPLACE
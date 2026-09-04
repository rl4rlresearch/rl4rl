MECHANISM: Preserve early anchors while ablating the next adjacent frame

HYPOTHESIS: Processing frames 3, 5, 6, 8–23, 27, and 29 will achieve at least 85% validation accuracy with 21 recurrent steps because it retains all early anchors implicated by prior failures while introducing only a single-frame gap before the dense command-bearing region.

INTENDED_EDIT: Remove frame 7 from the verified 22-step schedule while retaining frames 3, 5, and 6 plus the dense later region and both late samples.

EVIDENCE: Removing frame 5 reduced accuracy to 84.05%, while removing frame 6 narrowly missed at 84.91%; ablating frame 7 is the next direct test of whether those outcomes reflect specific early-frame importance rather than a general 21-step limit.

<<<<<<< SEARCH
                return schedule[:1] + schedule[2:]
=======
                return schedule[:1] + schedule[2:4] + schedule[5:]
>>>>>>> REPLACE
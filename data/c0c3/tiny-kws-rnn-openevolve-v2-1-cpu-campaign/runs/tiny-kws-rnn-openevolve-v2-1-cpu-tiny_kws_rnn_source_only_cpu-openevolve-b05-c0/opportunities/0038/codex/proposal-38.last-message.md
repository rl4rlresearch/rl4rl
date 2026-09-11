MECHANISM: Preserve early anchors while ablating an adjacent redundant frame

HYPOTHESIS: Processing frames 3, 5, 7–23, 27, and 29 will achieve at least 85% validation accuracy with 21 recurrent steps because it preserves frame 5 and avoids the two-frame early gap created by the failed frame-5 ablation.

INTENDED_EDIT: Remove frame 6 from the verified 22-step schedule, retaining frames 3 and 5 as early recurrent anchors plus the dense command-bearing region and both late samples.

EVIDENCE: The 22-step schedule using frames 3, 5–23, 27, and 29 achieved 85.03%, while removing frame 5 fell to 84.05%; testing frame 6 distinguishes whether that failure reflects frame 5’s specific importance or a general 21-step capacity limit.

<<<<<<< SEARCH
            if len(schedule) > 2:
                return schedule[:1] + schedule[2:]
            return schedule
=======
            if len(schedule) > 4:
                return schedule[:1] + schedule[2:3] + schedule[4:]
            return schedule
>>>>>>> REPLACE
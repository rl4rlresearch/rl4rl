MECHANISM: Late redundant-frame ablation

HYPOTHESIS: Removing frame 30 while preserving the passing schedule’s early frames and final endpoint will retain at least 85% validation accuracy while reducing recurrent steps and MACs by approximately 3.7%.

INTENDED_EDIT: Produce a 26-step schedule by dropping frame 30 from the verified 27-step schedule while retaining frame 31.

EVIDENCE: The 27-step schedule passed at 85.153%, but 26-step variants removing frames 1 or 2 failed; ablating a frame from the dense 28–31 tail tests whether early temporal coverage, rather than a strict 27-step requirement, caused those failures.

<<<<<<< SEARCH
        return schedule[1:] if len(schedule) > 2 else schedule
=======
        return schedule[1:-2] + schedule[-1:] if len(schedule) > 3 else schedule
>>>>>>> REPLACE
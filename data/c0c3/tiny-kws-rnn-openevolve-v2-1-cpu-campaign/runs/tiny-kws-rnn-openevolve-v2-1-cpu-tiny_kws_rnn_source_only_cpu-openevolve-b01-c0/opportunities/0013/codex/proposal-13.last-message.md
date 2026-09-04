MECHANISM: Alternative late-frame redundancy ablation

HYPOTHESIS: Removing frame 28 while preserving frames 29 and 31 will retain at least 85% validation accuracy and reduce recurrent steps and MACs by approximately 3.8% versus the passing 26-step design.

INTENDED_EDIT: Produce a 25-step schedule by dropping frame 28 from the verified 26-step schedule while retaining frame 29 and the final endpoint.

EVIDENCE: The 26-step schedule passed at 85.153%, whereas removing frame 29 fell to 84.540%; preserving frame 29 and ablating its neighboring frame 28 tests whether the failure was caused by losing that specific late observation rather than by the 25-step budget.

<<<<<<< SEARCH
        return schedule[1:-2] + schedule[-1:] if len(schedule) > 3 else schedule
=======
        return (
            schedule[1:-4] + schedule[-3:-2] + schedule[-1:]
            if len(schedule) > 4
            else schedule
        )
>>>>>>> REPLACE
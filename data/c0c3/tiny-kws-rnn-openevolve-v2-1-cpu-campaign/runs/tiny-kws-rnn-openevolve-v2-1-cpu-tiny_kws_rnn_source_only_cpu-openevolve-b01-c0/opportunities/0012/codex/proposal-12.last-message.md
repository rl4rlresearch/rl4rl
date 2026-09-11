MECHANISM: Consecutive late-frame ablation

HYPOTHESIS: Removing frame 29 while retaining frames 28 and 31 will preserve at least 85% validation accuracy and reduce recurrent steps and MACs by approximately 3.8% versus the passing 26-step design.

INTENDED_EDIT: Produce a 25-step schedule by dropping frame 29 from the verified 26-step schedule while preserving all earlier sampled frames and the final endpoint.

EVIDENCE: Removing frame 30 from the passing 27-step schedule retained 85.153% accuracy, indicating redundancy in the densely sampled 28–31 tail; frame 29 is the next controlled ablation in that region.

<<<<<<< SEARCH
        return schedule[1:-2] + schedule[-1:] if len(schedule) > 3 else schedule
=======
        return schedule[1:-3] + schedule[-1:] if len(schedule) > 4 else schedule
>>>>>>> REPLACE
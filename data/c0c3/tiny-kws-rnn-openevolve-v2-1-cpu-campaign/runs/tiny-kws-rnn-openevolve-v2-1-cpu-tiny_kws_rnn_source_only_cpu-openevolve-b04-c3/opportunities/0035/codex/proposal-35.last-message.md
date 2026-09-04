MECHANISM: Hierarchical recurrence with 22-step late-biased temporal thinning

HYPOTHESIS: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 883,789,260 to approximately 845,441,880.

INTENDED_EDIT: Reduce the schedule from 27 to 22 recurrent steps by dropping the six earliest frames from the original 28-frame grid, preserving the architecture and training procedure.

EVIDENCE: The identical hierarchy qualified at 85.77% with 23 steps and 883,789,260 MACs; testing the adjacent 22-step schedule directly locates the remaining temporal-compression boundary.

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE
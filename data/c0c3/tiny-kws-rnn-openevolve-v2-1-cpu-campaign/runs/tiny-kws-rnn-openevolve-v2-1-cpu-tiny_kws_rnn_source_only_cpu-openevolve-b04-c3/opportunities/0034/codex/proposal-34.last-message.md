MECHANISM: Hierarchical recurrence with 23-step late-biased temporal thinning

HYPOTHESIS: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 922,136,640 to approximately 883,789,260.

INTENDED_EDIT: Reduce the schedule to 23 recurrent steps by dropping the five earliest frames from the original 28-frame grid, preserving architecture and training procedure.

EVIDENCE: The identical hierarchy qualified at 86.26% with 24 steps and 922,136,640 MACs; testing the adjacent 23-step schedule directly locates the temporal-compression boundary.

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE
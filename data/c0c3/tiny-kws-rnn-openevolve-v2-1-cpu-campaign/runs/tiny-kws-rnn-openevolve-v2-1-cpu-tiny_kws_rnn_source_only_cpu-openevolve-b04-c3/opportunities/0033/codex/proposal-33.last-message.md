MECHANISM: Hierarchical recurrence with 24-step late-biased temporal thinning

HYPOTHESIS: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 960,484,020 to approximately 922,136,640.

INTENDED_EDIT: Reduce the current schedule from 25 to 24 recurrent steps by dropping the next-earliest frame from the original 28-frame grid.

EVIDENCE: The identical hierarchy achieved 88.22% at 26 steps and 88.47% at 25 steps, so the latest thinning preserved accuracy and leaves 3.47 percentage points of qualification margin.

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE
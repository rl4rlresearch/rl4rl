MECHANISM: Interior redundant-frame ablation

HYPOTHESIS: Keeping frame 1 while removing frame 2 from the passing 27-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.7%.

INTENDED_EDIT: Produce a 26-step schedule by preserving the passing schedule’s temporal endpoints and dropping the interior frame bracketed by adjacent frames 1 and 3.

EVIDENCE: Dropping frame 0 passed at 85.153%, while additionally dropping frame 1 narrowly failed at 84.908%; testing the neighboring but more redundant frame 2 isolates whether early coverage, rather than 26-step capacity, caused that miss.

<<<<<<< SEARCH
        return schedule[1:] if len(schedule) > 2 else schedule
=======
        if len(schedule) > 3:
            return schedule[1:2] + schedule[3:]
        return schedule
>>>>>>> REPLACE
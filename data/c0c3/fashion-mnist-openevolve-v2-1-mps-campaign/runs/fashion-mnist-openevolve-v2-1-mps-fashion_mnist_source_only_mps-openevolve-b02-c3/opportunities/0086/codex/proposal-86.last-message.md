MECHANISM: Mid-horizon full-state EMA refinement

HYPOTHESIS: An EMA decay of 0.98125 will exceed 9,344 correct predictions by interpolating within the accuracy-preserving region between 0.98 and 0.9825, without entering the degraded short-horizon regime observed at 0.9775.

INTENDED_EDIT: Change only the full-state EMA decay from 0.9825 to the untested midpoint 0.98125.

EVIDENCE: Decays 0.98 and 0.9825 both achieved 9,344 correct, while 0.9775 fell to 9,341; testing their midpoint most directly refines the remaining favorable EMA interval.

<<<<<<< SEARCH
                decay = 0.9825 ** elapsed
=======
                decay = 0.98125 ** elapsed
>>>>>>> REPLACE
MECHANISM: Five-step certified early exit

HYPOTHESIS: Beginning conservative classifier-bound checks at recurrent step 25 will retain validation accuracy at or above 85% while reducing total inference MACs below 778,346,514 and mean recurrent steps below 27.4283.

INTENDED_EDIT: Enable learned classification and the existing bounded-output exit certificate one recurrent step earlier, during the final five steps instead of the final four.

EVIDENCE: Extending certified checks from step 27 to step 26 reduced total MACs from 789,962,679 to 778,346,514, and label smoothing 0.02 restored accuracy to 85.15%; moving the unchanged certificate back one more step is the smallest supported cost-reduction test.

<<<<<<< SEARCH
        if bool(torch.all(count < 26.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 25.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 4)) & (
=======
        eligible = (completed >= float(total_steps - 5)) & (
>>>>>>> REPLACE
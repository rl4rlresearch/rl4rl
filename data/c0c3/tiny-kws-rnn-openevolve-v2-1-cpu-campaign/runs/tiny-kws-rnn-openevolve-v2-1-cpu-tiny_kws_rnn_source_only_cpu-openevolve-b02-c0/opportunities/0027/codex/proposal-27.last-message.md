MECHANISM: Four-step certified early exit

HYPOTHESIS: Beginning the exact classifier-bound checks at recurrent step 26 will preserve validation accuracy at or above 85% while reducing total inference MACs below 789,962,679 and mean recurrent steps below 27.8515.

INTENDED_EDIT: Enable the learned classifier and conservative bounded-output exit certificate during the final four recurrent steps instead of the final three.

EVIDENCE: Extending certified checks from step 28 to step 27 preserved 85.03% accuracy while reducing total MACs from 805,879,305 to 789,962,679; moving the same mathematically conservative certificate back one additional step is the smallest supported cost-reduction probe.

<<<<<<< SEARCH
        if bool(torch.all(count < 27.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 26.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 3)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 4)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE
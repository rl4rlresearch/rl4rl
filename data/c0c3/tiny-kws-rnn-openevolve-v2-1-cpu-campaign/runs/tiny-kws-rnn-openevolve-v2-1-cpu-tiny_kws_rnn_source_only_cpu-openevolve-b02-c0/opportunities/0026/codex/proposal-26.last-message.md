MECHANISM: Three-step certified early exit

HYPOTHESIS: Beginning the exact classifier-bound checks at recurrent step 27 will preserve validation accuracy at or above 85% while reducing total inference MACs and mean recurrent steps below the verified step-28 design.

INTENDED_EDIT: Enable the learned classifier and conservative bounded-output exit certificate during the final three recurrent steps instead of the final two.

EVIDENCE: Step-28 certified exit achieved 85.03% accuracy and reduced mean execution to 28.42 steps; extending the same mathematically conservative certificate by one step is the smallest next cost-reduction probe, while the much earlier step-20 attempt timed out without contrary accuracy evidence.

<<<<<<< SEARCH
        if bool(torch.all(count < 28.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 27.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 2)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 3)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE
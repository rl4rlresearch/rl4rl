MECHANISM: Eight-step certified early exit

HYPOTHESIS: Beginning conservative classifier-bound checks at recurrent step 22 will retain validation accuracy at or above 85% while reducing total inference MACs below 767,879,145 and mean recurrent steps below 27.0.

INTENDED_EDIT: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final eight steps instead of the final seven.

EVIDENCE: Extending the unchanged certificate successively through steps 26, 25, 24, and 23 preserved 85.15% accuracy while consistently reducing inference cost; step 23 achieved 767,879,145 MACs and exactly 27.0 mean steps.

<<<<<<< SEARCH
        if bool(torch.all(count < 23.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 22.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 7)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 8)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE
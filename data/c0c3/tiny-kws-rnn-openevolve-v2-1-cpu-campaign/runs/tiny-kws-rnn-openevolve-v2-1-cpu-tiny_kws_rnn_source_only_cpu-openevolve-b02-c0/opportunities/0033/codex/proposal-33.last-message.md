MECHANISM: Seven-step certified early exit

HYPOTHESIS: Beginning conservative classifier-bound checks at recurrent step 23 will retain validation accuracy at or above 85% while reducing total inference MACs below 768,548,682 and mean recurrent steps below 27.0454.

INTENDED_EDIT: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final seven steps instead of the final six.

EVIDENCE: Moving the certificate successively from step 26 through step 24 preserved 85.15% accuracy while reducing total MACs at every extension; step 24 reached 768,548,682 MACs and 27.0454 mean steps, motivating the smallest supported next probe.

<<<<<<< SEARCH
        if bool(torch.all(count < 24.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 23.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 6)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 7)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE
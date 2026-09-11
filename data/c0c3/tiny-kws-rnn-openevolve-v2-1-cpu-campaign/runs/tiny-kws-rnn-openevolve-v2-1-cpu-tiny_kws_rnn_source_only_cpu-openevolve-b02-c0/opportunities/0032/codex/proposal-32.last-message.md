MECHANISM: Six-step certified early exit

HYPOTHESIS: Beginning conservative classifier-bound checks at recurrent step 24 will retain validation accuracy at or above 85% while reducing total inference MACs below 771,635,007 and mean recurrent steps below 27.1743.

INTENDED_EDIT: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final six steps instead of the final five.

EVIDENCE: Moving the certificate from step 26 to step 25 with label smoothing 0.02 preserved 85.15% accuracy and reduced total MACs from 778,346,514 to 771,635,007; extending the unchanged certificate by one step is the smallest supported cost-reduction probe.

<<<<<<< SEARCH
        if bool(torch.all(count < 25.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 24.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 5)) & (
=======
        eligible = (completed >= float(total_steps - 6)) & (
>>>>>>> REPLACE
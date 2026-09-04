MECHANISM: Energy-screened ninth-step certified exit

HYPOTHESIS: Screening the highest-energy quartile at step 21 will produce at least five additional certified step savings, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,543,139.

INTENDED_EDIT: Start the existing energy-screened classifier and conservative exit certificate at step 21 instead of step 22, while retaining all later checks unchanged.

EVIDENCE: Moving the certificate earlier produced diminishing marginal savings of 105, 37, and 14 recurrent steps at steps 24, 23, and 22; the step-22 energy screen retained all 14 savings while evaluating only one quarter of examples, and five step-21 savings are sufficient to offset the added classifier calls.

<<<<<<< SEARCH
        if bool(torch.all(count < 22.0)):
            return averaged[:, :8]
        if bool(torch.all(count < 23.0)):
=======
        if bool(torch.all(count < 21.0)):
            return averaged[:, :8]
        if bool(torch.all(count < 23.0)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 8)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 9)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE
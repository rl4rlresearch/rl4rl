MECHANISM: Lower-side Adam second-moment refinement

HYPOTHESIS: AdamW beta2=0.989 will exceed 9,324 correct predictions by retaining the responsiveness of 0.99 while testing the still-unmeasured neighborhood above the degraded 0.98 setting.

INTENDED_EDIT: Change only AdamW’s second-moment decay from 0.99 to 0.989.

EVIDENCE: Beta2=0.99 achieved 9,324 correct, outperforming both 0.98 at 9,311 and 0.992 at 9,316; this brackets the optimum near 0.99 and makes a fine lower-side refinement the most informative next test.

<<<<<<< SEARCH
        betas=(0.9, 0.99),
=======
        betas=(0.9, 0.989),
>>>>>>> REPLACE
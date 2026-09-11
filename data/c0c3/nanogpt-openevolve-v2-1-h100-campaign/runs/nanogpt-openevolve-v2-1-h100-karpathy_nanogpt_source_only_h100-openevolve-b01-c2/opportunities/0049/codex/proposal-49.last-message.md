MECHANISM: Symmetric linear-schedule exposure reduction

HYPOTHESIS: Extending linear warmdown from 60% to 63% will retain roughly 520M training tokens while lowering val_bpb below 0.985181.

INTENDED_EDIT: Start the proven zero-ending linear decay three percentage points earlier, reducing integrated LR exposure by 0.015 peak-LR-time while preserving all peak learning rates and optimizer settings.

EVIDENCE: Adding a 5% LR floor increased exposure by the symmetric 0.015 and regressed val_bpb from 0.985181 to 0.985480; equal-area cosine also regressed, motivating a conservative exposure reduction that retains the successful linear shape and zero endpoint.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
=======
WARMDOWN_RATIO = 0.63   # begin linear LR warmdown after 37% of the time budget
>>>>>>> REPLACE
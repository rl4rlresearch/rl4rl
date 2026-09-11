MECHANISM: Training–evaluation spatial exposure alignment

HYPOTHESIS: Extending cyclic one-pixel translation training to 11/16 of the schedule will increase validation_correct above 9,256 by reducing the center-view training share from 60% to approximately 45%, matching the verified ensemble’s 44.8% center weight.

INTENDED_EDIT: Extend the existing five-view translation phase from one-half to eleven-sixteenths of training while preserving architecture, optimization, and evaluation.

EVIDENCE: Changing the verified 3.25:1 center-to-axis evaluation balance in either direction reduced validation_correct, while the current training schedule overexposes centered images relative to that balance; aligning their spatial proportions is the smallest training-side test of this evidence.

<<<<<<< SEARCH
    if step < total_steps // 2:
=======
    if step * 16 < total_steps * 11:
>>>>>>> REPLACE
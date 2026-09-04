MECHANISM: Center-biased terminal crop sampling

HYPOTHESIS: Restoring the verified 1/3 terminal centered-crop rate will reproduce 9,210 correct predictions, improving on the current 9,209-correct implementation.

INTENDED_EDIT: Increase terminal crop outcomes from five to six, adding a second centered outcome while preserving one outcome for each cardinal shift.

EVIDENCE: Reference Design 1 verified this exact change at 9,210 correct and 0.2253669 cross-entropy, the best qualified result; the current 1/5 centered rate reached 9,209, while increasing further to 3/7 fell to 9,208.

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE
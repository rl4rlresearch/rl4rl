MECHANISM: Alternate-coordinate token-position embedding gauge fixing

HYPOTHESIS: Fixing transfer feature 3 instead of the failed feature 6 will reduce the model to 1604 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the verified token-position quotient from features 4 and 5 to features 3, 4, and 5.

EVIDENCE: Features 4 and 5 achieved 0.9991 accuracy at 1605 parameters, whereas adding feature 6 fell to 0.3456; testing feature 3 isolates whether that failure was coordinate-specific.

<<<<<<< SEARCH
        self.transfer_features = (4, 5)
=======
        self.transfer_features = (3, 4, 5)
>>>>>>> REPLACE
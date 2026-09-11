MECHANISM: Endpoint-refined margin-adaptive flip fusion

HYPOTHESIS: Raising the bounded adjustment from 0.02984375 to 0.029921875 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962221206665039.

INTENDED_EDIT: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified setting and the unverified 0.03 endpoint.

EVIDENCE: Every verified increase from 0.025 through 0.02984375 preserved 9,331 correct predictions while monotonically reducing cross-entropy; the 0.03 attempts timed out without contrary validation evidence.

<<<<<<< SEARCH
                0.02984375
=======
                0.029921875
>>>>>>> REPLACE
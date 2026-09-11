MECHANISM: Feed-forward width boundary search

HYPOTHESIS: Setting `d_ff` to 11 will reduce the model from 1,644 to 1,627 learned parameters while preserving at least 99% accuracy.

INTENDED_EDIT: Change only the default feed-forward width from 12 to 11, preserving the successful architecture and training configuration otherwise.

EVIDENCE: The 1,644-parameter `d_ff=12` design achieved 99.96% accuracy, while `d_ff=10` collapsed to 11.92%; testing the intermediate width directly probes the apparent capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE
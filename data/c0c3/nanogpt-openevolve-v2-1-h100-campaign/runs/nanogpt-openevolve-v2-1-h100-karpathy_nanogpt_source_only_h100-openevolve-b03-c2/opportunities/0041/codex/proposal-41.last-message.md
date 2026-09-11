MECHANISM: Eighth-context local attention

HYPOTHESIS: Halving short-layer attention from 512 to 256 tokens while retaining two full-context layers and the verified schedules will reduce val_bpb below 0.984156 through greater token exposure within five minutes.

INTENDED_EDIT: Reduce only the six short attention windows from one-quarter to one-eighth of the 2048-token sequence length.

EVIDENCE: Reducing short attention from 1024 to 512 tokens increased exposure from 489.2M to 504.6M tokens and improved val_bpb from 0.985708 to 0.984156, motivating one further controlled halving.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE
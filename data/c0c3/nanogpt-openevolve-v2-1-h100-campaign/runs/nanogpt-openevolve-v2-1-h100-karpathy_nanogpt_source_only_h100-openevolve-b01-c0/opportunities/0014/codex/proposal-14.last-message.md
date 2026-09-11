MECHANISM: Delayed linear learning-rate cooldown

HYPOTHESIS: Shortening warmdown from 50% to 40% will beat 0.985318 val_bpb by providing more cumulative optimization at the proven 262,144-token batch without increasing memory use or altering throughput.

INTENDED_EDIT: Preserve the best architecture and batch configuration while delaying the start of the linear decay-to-zero schedule.

EVIDENCE: The 112-sequence run with all learning rates reduced by 7/8 regressed from 0.985719 to 0.986515, indicating that less cumulative parameter movement was unhelpful; a modestly shorter warmdown tests the opposite direction at the best-performing 128-sequence configuration.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
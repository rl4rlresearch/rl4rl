MECHANISM: Nineteen-frame endpoint-aware recurrent readout

HYPOTHESIS: The qualified 120-unit dual-readout GRU will retain at least 85% validation accuracy on 19 uniformly distributed frames while reducing total inference MACs from 823,084,800 to approximately 782,008,800 and recurrent steps from 20 to 19 per example.

INTENDED_EDIT: Reduce the uniform causal frame schedule from 20 frames to 19 while preserving recurrent capacity, readouts, and training procedure.

EVIDENCE: The current 120-unit, 20-frame design achieved 86.26% accuracy, providing a 1.26-point margin; removing one of its uniformly distributed frames yields a larger structural cost reduction than the riskier width reduction below the barely qualifying 118-unit design.

<<<<<<< SEARCH
        steps = min(20, available_frames)
=======
        steps = min(19, available_frames)
>>>>>>> REPLACE
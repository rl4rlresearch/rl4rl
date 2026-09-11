MECHANISM: Redundant early-frame pruning

HYPOTHESIS: Removing one additional densely spaced early frame from the 83-unit single-gate model will preserve at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and total inference MACs by approximately 3.8%.

INTENDED_EDIT: Expand the existing schedule deletion from two to three early indices, preserving the recurrent architecture, temporal readout, training procedure, and full recording coverage.

EVIDENCE: The current 83-unit single-gate model achieved 86.26% accuracy at 26 steps, leaving 1.26 percentage points of margin; its schedule already benefits from removing two adjacent early frames, motivating a controlled test that removes one more nearby frame while retaining the first and final frames.

<<<<<<< SEARCH
        if len(indices) > 27:
            del indices[1:3]
=======
        if len(indices) > 27:
            del indices[1:4]
>>>>>>> REPLACE
MECHANISM: Uniform temporal-grid coarsening

HYPOTHESIS: Re-spacing the 117-unit GRU’s coverage onto a uniform 23-step grid will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.2% versus the passing 24-step model.

INTENDED_EDIT: Reduce the base interpolation grid from 28 to 27 points; the existing slicing then executes 23 uniformly redistributed causal frames while retaining the final frame.

EVIDENCE: The 24-step model passed at 85.153%, while deleting a late frame scored 84.294% and deleting the second frame narrowly missed at 84.908%; redistributing all intermediate samples tests whether temporal spacing, rather than 23-step capacity, caused those failures.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(27, available_frames)
>>>>>>> REPLACE
MECHANISM: Nested trailing-edge frame omission

HYPOTHESIS: Removing the penultimate input frame from the qualified dual-readout model will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.57%.

INTENDED_EDIT: Reduce the causal schedule from 28 to 27 steps by omitting frame 30 in addition to frame 1, while preserving both endpoints, every interior selected frame, and the 111-unit dual-view readout.

EVIDENCE: The current 28-step dual-readout design achieved 87.48% accuracy, a 2.48-point margin; moreover, the earlier single-readout transition from 29 steps to the nested 28-step schedule lost only 0.245 points, supporting an isolated additional edge-adjacent omission.

<<<<<<< SEARCH
        return [frame for frame in schedule if frame != 1]
=======
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
            ]
        return schedule
>>>>>>> REPLACE
MECHANISM: Nested early-edge frame subsampling

HYPOTHESIS: Omitting frame 2 from the qualified 110-unit, 24-step schedule while retaining the global-mean/final-state readout will preserve at least 85% accuracy and reduce total inference MACs by approximately 4.16%.

INTENDED_EDIT: Reduce execution to 23 recurrent steps by additionally excluding frame index 2, without changing the qualified architecture, readout, or training procedure.

EVIDENCE: The qualified 24-step model achieved 85.40%; previously removing adjacent early frame 1 cost only 0.245 points, while the failed 23-step experiment also changed the readout and removed late frame 25, so an isolated early-frame removal is the most informative lower-cost test.

<<<<<<< SEARCH
                not in (
                    1,
                    available_frames - 5,
=======
                not in (
                    1,
                    2,
                    available_frames - 5,
>>>>>>> REPLACE
MECHANISM: Intermediate-minibatch update-density interpolation

HYPOTHESIS: Batch size 160 will exceed 9,268 correct predictions by gaining more optimizer updates than batch 192 without incurring the accuracy loss observed at batch 128.

INTENDED_EDIT: Restore the best mildly distance-stratified radius-2 TTA and reduce training batch size from 192 to 160; preserve the architecture, optimizer, loss, schedule, and calibration.

EVIDENCE: Batch 192 achieved 9,268 correct, while batch 128 improved cross-entropy from 0.21246 to 0.20833 but lost five predictions; batch 160 tests the unmeasured interval between their optimization regimes.

<<<<<<< SEARCH
BATCH_SIZE = 192
=======
BATCH_SIZE = 160
>>>>>>> REPLACE

<<<<<<< SEARCH
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
=======
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE
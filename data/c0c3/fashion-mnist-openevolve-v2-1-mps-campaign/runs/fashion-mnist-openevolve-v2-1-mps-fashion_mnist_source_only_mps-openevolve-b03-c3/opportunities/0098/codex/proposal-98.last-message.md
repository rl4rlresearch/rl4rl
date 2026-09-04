MECHANISM: Conservative vertical-axis TTA interpolation

HYPOTHESIS: Vertical/horizontal radius-1 weights of 0.1025/0.0775 will retain 9,268 correct predictions while lowering cross-entropy below 0.2124414, strictly improving validation_score.

INTENDED_EDIT: Extend the successful vertical TTA bias by one quarter of the tested 0.10-to-0.11 interval, preserving total ensemble weight and all training behavior.

EVIDENCE: The 0.10/0.08 weighting retained 9,268 correct and improved cross-entropy to 0.2124414; 0.11/0.07 further reduced cross-entropy to 0.2124401 but lost five predictions, motivating a cautious interpolation near the accuracy-preserving setting.

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.1025
                    elif delta_y == 0:
                        weight = 0.0775
                    else:
                        weight = 0.045
>>>>>>> REPLACE
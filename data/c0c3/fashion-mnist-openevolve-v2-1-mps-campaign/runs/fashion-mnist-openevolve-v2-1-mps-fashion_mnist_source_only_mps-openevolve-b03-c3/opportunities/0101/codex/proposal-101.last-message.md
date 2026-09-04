MECHANISM: Opposite sign-aware vertical translation ensembling

HYPOTHESIS: Favoring the downward-content radius-1 view over the upward-content view will exceed 9,268 correct predictions because the previously tested opposite asymmetry materially changed predictions and reduced cross-entropy, demonstrating useful vertical sign sensitivity but choosing the wrong directional preference.

INTENDED_EDIT: Preserve the accuracy-safe total vertical TTA weight of 0.20 while assigning 0.11 to the downward-content shift and 0.09 to the upward-content shift; retain all other architecture, training, and TTA settings.

EVIDENCE: Symmetric 0.10/0.10 vertical weighting achieved 9,268 correct, while favoring the upward-content view produced 9,264 with substantially lower 0.2124019 cross-entropy; testing the complementary direction is the cleanest unresolved sign-aware comparison.

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.11 if delta_y < 0 else 0.09
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
>>>>>>> REPLACE
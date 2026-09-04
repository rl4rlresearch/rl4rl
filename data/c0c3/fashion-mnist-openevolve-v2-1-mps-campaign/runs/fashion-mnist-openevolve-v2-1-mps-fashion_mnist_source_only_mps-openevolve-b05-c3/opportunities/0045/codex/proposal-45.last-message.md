MECHANISM: Boundary-complete overlapping max pooling

HYPOTHESIS: Symmetric final-stage pooling will exceed 9,167 correct predictions by improving one-pixel translation consistency without discarding the position-sensitive 3×3 feature grid.

INTENDED_EDIT: Replace the final 2×2 stride-2 pool, which ignores the last feature row and column of the 7×7 map, with overlapping 3×3 stride-2 pooling that covers the complete map.

EVIDENCE: The 39/64 translation curriculum achieved 9,167 correct, while global attention pooling fell to 9,103; this motivates retaining localized spatial pooling while removing its asymmetric boundary truncation.

<<<<<<< SEARCH
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
=======
            if index in (1, 3):
                layers.append(nn.MaxPool2d(2))
            elif index == 5:
                layers.append(nn.MaxPool2d(3, stride=2))
>>>>>>> REPLACE
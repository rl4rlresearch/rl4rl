MECHANISM: Flip-paired trimmed translation-logit pooling

HYPOTHESIS: Trimming the highest and lowest translation logits per class after flip-pair averaging will exceed 9,287 correct predictions by suppressing direction-specific padding artifacts while retaining evidence from three translation positions.

INTENDED_EDIT: Replace the uniform mean over all ten evaluation logits with flip-pair averaging followed by a coordinate-wise trimmed mean across the five translations; training remains unchanged.

EVIDENCE: Uniform ten-view logit averaging produced the best result at 9,287 correct, while center-prioritized pooling timed out; robust trimming tests the same suspected boundary-artifact mechanism without assuming the center view is always best.

<<<<<<< SEARCH
        return view_logits.mean(dim=(0, 1))
=======
        position_logits = view_logits.mean(dim=1)
        sorted_logits = position_logits.sort(dim=0).values
        return sorted_logits[1:-1].mean(dim=0)
>>>>>>> REPLACE
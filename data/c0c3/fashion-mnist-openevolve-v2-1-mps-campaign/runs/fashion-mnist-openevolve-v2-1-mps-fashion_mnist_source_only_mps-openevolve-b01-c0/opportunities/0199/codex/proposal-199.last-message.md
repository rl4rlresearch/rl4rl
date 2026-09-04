MECHANISM: Float64 sub-ULP temperature interpolation

HYPOTHESIS: Evaluating logits in float64 at 0.7176630539553506 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Cast the final ensemble logits to float64 and use the quadratic minimum approximately one-fourteenth of a float32 ULP above the current temperature.

EVIDENCE: The adjacent float32 temperatures both worsened cross-entropy; the upper neighbor increased it by three reporting units and the lower by four, implying a continuous quadratic minimum about 1/14 ULP above the current value that float32 division cannot represent.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities.double() / 0.7176630539553506
>>>>>>> REPLACE
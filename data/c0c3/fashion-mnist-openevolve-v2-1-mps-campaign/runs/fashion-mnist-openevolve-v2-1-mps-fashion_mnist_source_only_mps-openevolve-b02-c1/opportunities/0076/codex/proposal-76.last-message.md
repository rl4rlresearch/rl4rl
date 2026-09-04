MECHANISM: Center-biased test-time augmentation fusion

HYPOTHESIS: Increasing each centered view’s weight from 2.0 to 2.25 will exceed 9,286 correct predictions by reducing boundary-artifact influence from translated views without increasing runtime or parameters.

INTENDED_EDIT: Give the original and horizontally flipped centered views slightly more ensemble weight, update the normalization constant, and retain the proven temperature.

EVIDENCE: Temperature calibration has plateaued at 9,286 correct, while representation-changing attempts timed out; reweighting the existing 12-view ensemble is the lowest-cost mechanism that can change argmax predictions.

<<<<<<< SEARCH
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
=======
        center_weight = 2.25
        ensemble = F.log_softmax(logits, dim=1) + math.log(center_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
=======
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(center_weight)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.5)) / 0.912
>>>>>>> REPLACE
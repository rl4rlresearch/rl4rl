MECHANISM: Stronger center-prior test-time ensemble

HYPOTHESIS: Increasing each unshifted view’s ensemble weight from 2 to 3 will exceed 9,280 correct predictions by favoring the validation images’ native alignment while retaining shifted-view robustness.

INTENDED_EDIT: Reweight the existing probability ensemble so centered original and flipped predictions each receive weight 3, with shifted predictions unchanged.

EVIDENCE: The proven 9,280-correct design already privileges centered views over shifted views; recent training-side changes either timed out or reduced accuracy, motivating a conservative, computation-neutral refinement of that established ensemble.

<<<<<<< SEARCH
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
=======
        ensemble = F.log_softmax(logits, dim=1) + math.log(3.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return ensemble - math.log(12.0)
=======
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(3.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return ensemble - math.log(14.0)
>>>>>>> REPLACE
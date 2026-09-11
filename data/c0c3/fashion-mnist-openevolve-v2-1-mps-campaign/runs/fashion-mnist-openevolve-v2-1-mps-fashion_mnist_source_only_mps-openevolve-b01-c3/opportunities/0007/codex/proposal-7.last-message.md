MECHANISM: Center-weighted translation-and-flip logit ensemble

HYPOTHESIS: Giving the centered flip pair twice the weight of each cardinally shifted pair will exceed 9,111 correct predictions by retaining the useful translation ensemble while reducing distortion from shifted views.

INTENDED_EDIT: Keep the validated architecture, training recipe, and ten evaluation views, but replace uniform logit averaging with a 2:1 center-to-cardinal weighting.

EVIDENCE: Cardinal translation averaging improved flip-only inference from 9,094 to 9,111 correct, while adding equally weighted diagonal translations reduced performance to 9,109, indicating that shifted views help but excessive uniform weighting can dilute the most reliable centered prediction.

<<<<<<< SEARCH
        return self._predict(torch.cat(views, dim=0)).reshape(
            len(views), images.shape[0], 10
        ).mean(dim=0)
=======
        logits = self._predict(torch.cat(views, dim=0)).reshape(
            len(views), images.shape[0], 10
        )
        weights = logits.new_tensor(
            (2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        ).view(-1, 1, 1)
        return (logits * weights).sum(dim=0) / weights.sum()
>>>>>>> REPLACE
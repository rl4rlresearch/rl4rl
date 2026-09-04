MECHANISM: Reduced tail averaging with bounded batched TTA

HYPOTHESIS: Reducing tail-average interpolation from 0.5 to 0.25 will reverse the boundary drift observed at 0.75 and achieve at least 9,287 correct predictions; batching the nine auxiliary views into two memory-bounded calls will complete within the time limit.

INTENDED_EDIT: Give the late-checkpoint average 25% final weight and replace nine sequential auxiliary-view evaluations with batches of four unflipped and five flipped views while preserving fusion order, weights, and temperature.

EVIDENCE: Increasing tail interpolation to 0.75 reduced validation-correct from 9,286 to 9,276, directly motivating movement below 0.5. Earlier 0.25 attempts timed out without accuracy evidence, while bounded batched probability TTA has completed successfully.

<<<<<<< SEARCH
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
=======
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        batch = images.shape[0]

        translated = torch.cat(views[1:], dim=0)
        translated_logits = self.classifier(self.features(translated))
        for view_logits in translated_logits.split(batch, dim=0):
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )

        flipped = torch.cat([view.flip(-1) for view in views], dim=0)
        flipped_logits = self.classifier(self.features(flipped))
        for view_index, view_logits in enumerate(
            flipped_logits.split(batch, dim=0)
        ):
            flipped_log_probs = F.log_softmax(view_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.25)
>>>>>>> REPLACE
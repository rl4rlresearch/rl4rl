MECHANISM: Batched vertical-shift probability ensembling

HYPOTHESIS: Adding one-pixel up/down views to the hard-maximum model’s validation ensemble will exceed 9,322 correct predictions by reducing pooling-phase sensitivity, while arithmetic probability averaging preserves the previously observed cross-entropy advantage over mean-logit ensembling.

INTENDED_EDIT: During evaluation, jointly infer the original, horizontal-flip, one-pixel-up, and one-pixel-down views in both orientations, then average their class probabilities.

EVIDENCE: The hard-maximum design reliably finishes near 78 seconds with 9,320 correct, and arithmetic flip averaging achieved lower cross-entropy than geometric averaging at the same count; the attempted training-time translation experiment timed out, motivating translation diversity only during validation.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (0, 0, 1, 1), mode="replicate")
        shifted_up = padded[:, :, 2:, :]
        shifted_down = padded[:, :, :-2, :]
        spatial_views = torch.cat(
            (images, shifted_up, shifted_down),
            dim=0,
        )
        all_views = torch.cat(
            (spatial_views, torch.flip(spatial_views, dims=(-1,))),
            dim=0,
        )
        view_logits = self._forward_once(all_views)
        view_logits = view_logits.reshape(6, images.shape[0], 10)
        log_probabilities = F.log_softmax(view_logits, dim=2)
        return torch.logsumexp(log_probabilities, dim=0) - math.log(6.0)
>>>>>>> REPLACE
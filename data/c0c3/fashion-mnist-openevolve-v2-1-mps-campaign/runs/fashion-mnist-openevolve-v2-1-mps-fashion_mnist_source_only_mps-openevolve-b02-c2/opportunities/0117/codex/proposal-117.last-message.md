MECHANISM: Batched calibrated top-four ensemble

HYPOTHESIS: Preserving the 9,322-correct top-four model while applying temperature 1.05 will lower validation cross-entropy below 0.192551; unsorted selection, fused attention, and batched flip inference will reduce timeout risk.

INTENDED_EDIT: Keep exact top-four saliency, disable unnecessary top-k sorting, fuse the bias-free channel-attention calls, evaluate both flip views in one forward pass, and mildly soften the probability-mixture logits.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322, while arithmetic probability averaging gave the lowest hard-maximum cross-entropy without changing its predictions; repeated top-four calibration attempts timed out, motivating algebraically equivalent runtime reductions.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        paired_images = torch.cat(
            (images, torch.flip(images, dims=(-1,))),
            dim=0,
        )
        logits, flipped_logits = self._forward_once(paired_images).chunk(
            2, dim=0
        )
        mixture_logits = torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
        return mixture_logits / 1.05
>>>>>>> REPLACE
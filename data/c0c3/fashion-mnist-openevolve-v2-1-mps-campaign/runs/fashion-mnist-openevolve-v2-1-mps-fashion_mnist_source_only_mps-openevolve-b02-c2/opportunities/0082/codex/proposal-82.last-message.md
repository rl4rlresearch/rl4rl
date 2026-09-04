MECHANISM: Calibrated unsorted top-four channel saliency

HYPOTHESIS: Unsorted top-four attention will retain the best design’s 9,322 correct predictions, while temperature 1.05 lowers its validation cross-entropy and reduced top-k ordering work improves completion reliability.

INTENDED_EDIT: Restore top-four channel evidence using unsorted selection and mildly soften only the arithmetic flip-mixture logits at validation.

EVIDENCE: Reference Design 2 achieved the best verified count with top-four saliency; arithmetic flip ensembling produced lower cross-entropy than geometric ensembling, motivating class-preserving temperature calibration.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = features.flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        mixture_logits = (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        )
        return mixture_logits / 1.05
>>>>>>> REPLACE
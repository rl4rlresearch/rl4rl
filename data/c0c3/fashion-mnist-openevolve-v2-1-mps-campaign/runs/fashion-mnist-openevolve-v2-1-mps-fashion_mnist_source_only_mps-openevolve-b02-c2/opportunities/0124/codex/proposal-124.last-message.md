MECHANISM: Asymmetric paired saliency co-training

HYPOTHESIS: Applying exact top-four saliency to one view of every training pair and hard-maximum saliency to the other will finish within the time limit and exceed 9,322 correct predictions by preserving per-example top-four exposure while regularizing the shared attention kernel; arithmetic probability ensembling will reduce cross-entropy when counts tie.

INTENDED_EDIT: Use top-four channel evidence for the first paired view and inexpensive hard-maximum evidence for its flipped counterpart during training, use top-four evidence for both validation views, fuse the shared bias-free attention calls, and return an arithmetic probability ensemble.

EVIDENCE: Full top-four training achieved the best verified count of 9,322 but cost 78.8 seconds, while validation-only top-four regressed to 9,315, showing that training exposure is necessary. Hard-maximum training reliably finished near 9,320, and arithmetic probability averaging previously improved its cross-entropy without changing predictions.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        flattened = features.flatten(2)
        if self.training:
            paired_split = flattened.shape[0] // 2
            top_four = flattened[:paired_split].topk(
                4, dim=2, sorted=False
            ).values.mean(dim=2, keepdim=True)
            maximum = flattened[paired_split:].amax(dim=2, keepdim=True)
            channel_salient = torch.cat((top_four, maximum), dim=0)
        else:
            channel_salient = flattened.topk(
                4, dim=2, sorted=False
            ).values.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE
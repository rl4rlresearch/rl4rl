MECHANISM: Slightly stronger constant label smoothing with reliable hard-max attention

HYPOTHESIS: Holding label smoothing at 0.025 throughout training will exceed 9,328 correct predictions by strengthening the regularization benefit observed at 0.02 without the late hard-target transition that reduced accuracy.

INTENDED_EDIT: Restore the runtime-reliable hard-maximum channel descriptor and 0.10 confidence-adaptive flip fusion, then train paired views with constant 0.025 label smoothing.

EVIDENCE: Constant 0.02 smoothing improved hard-max attention from 9,320 to 9,328 correct, while annealing smoothing toward zero reduced the result to 9,325; this motivates a small upward search around the best verified constant setting.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        top_two = log_probabilities.topk(2, dim=2).values
        margins = top_two[..., 0] - top_two[..., 1]
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.025,
    )
>>>>>>> REPLACE
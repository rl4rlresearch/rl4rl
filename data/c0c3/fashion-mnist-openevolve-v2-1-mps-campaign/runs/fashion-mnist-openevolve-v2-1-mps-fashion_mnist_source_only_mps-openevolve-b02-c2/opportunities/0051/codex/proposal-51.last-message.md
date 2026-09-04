MECHANISM: Stop-gradient top-four channel saliency

HYPOTHESIS: Top-four saliency with its sparse feature-path gradient detached will exceed 9,322 correct predictions while finishing within the time limit by preserving the best verified attention signal and eliminating top-k backward propagation.

INTENDED_EDIT: Replace strongest-local-patch saliency with top-four averaging computed from detached features; retain gradients through the attention kernel, gate, and main feature path.

EVIDENCE: Exact top-four saliency achieved the best result at 9,322 correct, but subsequent top-k implementations timed out; the faster local-patch approximation finished but regressed to 9,312, motivating preservation of exact top-four values with a cheaper, less disruptive backward pass.

<<<<<<< SEARCH
        channel_salient = F.avg_pool2d(features, kernel_size=2, stride=1)
        channel_salient = F.adaptive_max_pool2d(channel_salient, 1)
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
=======
        channel_salient = features.detach().flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
>>>>>>> REPLACE
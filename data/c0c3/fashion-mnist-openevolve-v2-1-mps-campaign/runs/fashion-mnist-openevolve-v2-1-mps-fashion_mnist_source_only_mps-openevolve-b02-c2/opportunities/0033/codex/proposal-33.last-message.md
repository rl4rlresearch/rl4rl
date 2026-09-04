MECHANISM: Peak-biased top-four channel saliency

HYPOTHESIS: Blending 25% hard-maximum evidence into the top-four descriptor will exceed 9,322 correct predictions by preserving top-four robustness while recovering useful peak emphasis.

INTENDED_EDIT: Reweight the existing top-four activations toward their strongest member without adding parameters or another top-k operation.

EVIDENCE: Top-four averaging achieved 9,322 correct, while hard maximum achieved 9,320 with slightly lower cross-entropy (0.192262 versus 0.192551), motivating a conservative interpolation between them.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
=======
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = (
            0.75 * channel_salient.mean(dim=2, keepdim=True)
            + 0.25 * channel_salient[:, :, :1]
        )
        channel_salient = channel_salient.transpose(1, 2)
>>>>>>> REPLACE
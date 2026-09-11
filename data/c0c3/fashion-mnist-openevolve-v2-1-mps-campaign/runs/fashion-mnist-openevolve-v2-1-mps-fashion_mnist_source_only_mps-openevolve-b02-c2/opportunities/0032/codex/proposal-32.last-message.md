MECHANISM: Softmax-weighted channel saliency

HYPOTHESIS: Replacing normalized log-mean-exp with a temperature-0.25 softmax-weighted activation mean will exceed 9,322 correct predictions by retaining dense gradients while restoring the peak-scale descriptor lost through log-mean-exp normalization.

INTENDED_EDIT: Use the Gibbs-weighted mean of spatial activations as the salient channel descriptor, preserving architecture, parameters, training, and flip-ensemble inference.

EVIDENCE: Top-four saliency achieved 9,322 correct and hard maxima achieved 9,320, while normalized log-mean-exp fell to 9,301; softmax weighting preserves the current efficient dense selection but avoids the normalization-induced downward shift in descriptor magnitude.

<<<<<<< SEARCH
        spatial_features = features.flatten(2)
        temperature = 0.25
        channel_salient = temperature * (
            torch.logsumexp(spatial_features / temperature, dim=2, keepdim=True)
            - math.log(spatial_features.shape[2])
        )
        channel_salient = channel_salient.transpose(1, 2)
=======
        spatial_features = features.flatten(2)
        temperature = 0.25
        salient_weights = F.softmax(spatial_features / temperature, dim=2)
        channel_salient = (spatial_features * salient_weights).sum(
            dim=2, keepdim=True
        )
        channel_salient = channel_salient.transpose(1, 2)
>>>>>>> REPLACE
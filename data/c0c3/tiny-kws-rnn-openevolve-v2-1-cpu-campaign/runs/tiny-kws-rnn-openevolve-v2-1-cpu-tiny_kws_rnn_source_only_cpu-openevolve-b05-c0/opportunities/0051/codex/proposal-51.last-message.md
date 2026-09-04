MECHANISM: Orthonormal zero-sum logit parameterization

HYPOTHESIS: A seven-output classifier mapped through fixed orthonormal Helmert contrasts will retain at least 85% accuracy while saving 114 MACs per example and 115 parameters.

INTENDED_EDIT: Replace the 114-to-8 classifier with a 114-to-7 classifier and transform its outputs into eight zero-sum logits using elementwise operations.

EVIDENCE: The fixed-reference seven-logit design failed at 83.93% despite preserving the full softmax family; an orthonormal, class-balanced contrast basis targets its asymmetric optimization geometry without restoring the redundant eighth learned output.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)
=======
        self.classifier = nn.Linear(114, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
=======
        contrasts = self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
        scales = contrasts.new_tensor(
            [math.sqrt((index + 1) * (index + 2)) for index in range(7)]
        )
        scaled = contrasts / scales
        tails = torch.flip(
            torch.cumsum(torch.flip(scaled, dims=(-1,)), dim=-1), dims=(-1,)
        )
        middle = tails[:, 1:] - scaled[:, :-1] * scaled.new_tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        return torch.cat(
            (tails[:, :1], middle, -7.0 * scaled[:, -1:]), dim=-1
        )
>>>>>>> REPLACE
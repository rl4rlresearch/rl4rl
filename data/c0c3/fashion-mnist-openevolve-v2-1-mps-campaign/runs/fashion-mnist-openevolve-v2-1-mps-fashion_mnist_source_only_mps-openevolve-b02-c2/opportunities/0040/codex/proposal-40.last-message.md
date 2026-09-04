MECHANISM: Dimension-selective AdamW regularization

HYPOTHESIS: Restoring top-four channel saliency and plain paired-view cross-entropy while excluding normalization and bias vectors from weight decay will exceed 9,322 correct predictions by preserving the strongest verified architecture and avoiding unnecessary shrinkage of calibration parameters.

INTENDED_EDIT: Use the verified top-four descriptor and paired-view loss, then apply AdamW decay only to multidimensional kernel and matrix parameters.

EVIDENCE: Top-four attention with plain paired-view cross-entropy achieved the best result at 9,322 correct, while ensemble-aware training fell to 9,307; selective decay is an orthogonal, parameter-neutral optimizer refinement.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay_parameters = []
    calibration_parameters = []
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            calibration_parameters.append(parameter)
    return torch.optim.AdamW(
        (
            {"params": decay_parameters, "weight_decay": 5e-4},
            {"params": calibration_parameters, "weight_decay": 0.0},
        ),
        lr=2.5e-3,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    ensemble_log_probabilities = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(paired_logits[:batch_size], dim=1),
                F.log_softmax(paired_logits[batch_size:], dim=1),
            )
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * (individual_loss + ensemble_loss)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE
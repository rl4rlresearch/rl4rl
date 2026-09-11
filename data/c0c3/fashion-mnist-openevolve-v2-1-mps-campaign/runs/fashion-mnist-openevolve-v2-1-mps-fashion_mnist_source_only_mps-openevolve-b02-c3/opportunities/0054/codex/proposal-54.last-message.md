MECHANISM: Late hard-label sharpening on dual-statistic refinement gating

HYPOTHESIS: Restoring the best mean/max gate and annealing label smoothing to zero during the final 20% of optimization will exceed 9,322 correct predictions by reducing smoothing bias during low-learning-rate convergence.

INTENDED_EDIT: Restore Reference Design 3’s equal mean/max gate descriptor and progressively reduce label smoothing from 0.02 to zero over the last fifth of training.

EVIDENCE: Reference Design 3 achieved the best result at 9,322 correct; later architectural, gate-capacity, and flip-ensemble-loss changes regressed, motivating a conservative optimization refinement of that proven model.

<<<<<<< SEARCH
        pooled = F.adaptive_avg_pool2d(refined, output_size=1)
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
=======
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    smoothing = 0.02 * min((1.0 - progress) / 0.20, 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=smoothing,
    )
>>>>>>> REPLACE
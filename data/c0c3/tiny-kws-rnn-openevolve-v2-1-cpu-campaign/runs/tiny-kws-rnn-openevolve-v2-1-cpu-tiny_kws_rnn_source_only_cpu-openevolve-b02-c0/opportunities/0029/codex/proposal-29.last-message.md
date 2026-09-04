MECHANISM: Verifiable late-prefix deep supervision

HYPOTHESIS: Supervising the four summaries immediately preceding the final state will recover the one-example accuracy deficit of step-26 certified exit, achieving at least 85% accuracy while reducing MACs below the verified step-27 design.

INTENDED_EDIT: Cache steps 26–29 during the standard training sequence, apply a lightly weighted auxiliary classification loss to them, and enable certified exits beginning at step 26.

EVIDENCE: Step-26 certified exit reached 84.91%, only one validation example below the requirement, while reducing total inference MACs to 778,595,301; targeted prefix supervision directly trains the representations used by those earlier checks without changing inference architecture or parameter count.

<<<<<<< SEARCH
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
        self._training_prefixes: torch.Tensor | None = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
        outputs, hidden = self.gru(
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
        )
        if self.training and frames.shape[1] >= 5:
            prefix_sums = summary.unsqueeze(1) + outputs.cumsum(dim=1)
            prefix_counts = count.unsqueeze(1) + torch.arange(
                1,
                frames.shape[1] + 1,
                device=frames.device,
                dtype=frames.dtype,
            ).view(1, -1, 1)
            self._training_prefixes = (
                prefix_sums / prefix_counts.clamp_min(1.0)
            )[:, -5:-1, :]
        else:
            self._training_prefixes = None
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if bool(torch.all(count < 27.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 26.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = (completed >= float(total_steps - 3)) & (
            completed < float(total_steps)
        )
=======
        eligible = (completed >= float(total_steps - 4)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    del step, total_steps
    loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    prefixes = getattr(model, "_training_prefixes", None)
    if prefixes is None:
        return loss

    model._training_prefixes = None
    prefix_logits = model.classifier(prefixes.flatten(0, 1))
    prefix_labels = labels[:, None].expand(
        -1, prefixes.shape[1]
    ).reshape(-1)
    prefix_loss = F.cross_entropy(
        prefix_logits,
        prefix_labels,
        label_smoothing=0.03,
    )
    return loss + 0.05 * prefix_loss
>>>>>>> REPLACE
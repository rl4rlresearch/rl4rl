MECHANISM: Late-prefix deep supervision with four-step certified exit

HYPOTHESIS: Supervising classifier predictions at recurrent steps 26–29 will preserve final validation accuracy at or above 85% while enabling certified checks from step 26 to reduce MACs below the verified step-27 design.

INTENDED_EDIT: Cache training-only logits from the four late prefixes, add a lightly weighted auxiliary cross-entropy loss, and begin learned classification and certified exit checks at step 26.

EVIDENCE: Step-26 certified exit missed the threshold by one validation example at 84.91% while lowering total MACs to 778,595,301; step-27 met the target at 85.03%, so targeted late-prefix supervision is a minimal accuracy-oriented change to recover that narrow deficit.

<<<<<<< SEARCH
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
        self._auxiliary_logits: torch.Tensor | None = None
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
        self._auxiliary_logits = None
        if self.training and outputs.shape[1] >= 29:
            cumulative = summary[:, None, :] + outputs.cumsum(dim=1)
            prefix_counts = count[:, None, :] + torch.arange(
                1,
                outputs.shape[1] + 1,
                device=outputs.device,
                dtype=outputs.dtype,
            )[None, :, None]
            late_averages = (
                cumulative[:, 25:29, :] / prefix_counts[:, 25:29, :]
            )
            self._auxiliary_logits = self.classifier(late_averages)
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
    auxiliary_logits = getattr(model, "_auxiliary_logits", None)
    if auxiliary_logits is not None:
        auxiliary_labels = labels[:, None].expand(
            -1, auxiliary_logits.shape[1]
        )
        loss = loss + 0.20 * F.cross_entropy(
            auxiliary_logits.reshape(-1, auxiliary_logits.shape[-1]),
            auxiliary_labels.reshape(-1),
            label_smoothing=0.03,
        )
    return loss
>>>>>>> REPLACE
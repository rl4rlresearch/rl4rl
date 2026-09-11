MECHANISM: Deep-supervised adjacent-width reduction

HYPOTHESIS: A 121-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 122-unit design.

INTENDED_EDIT: Reduce recurrent and readout width to 121 units and add auxiliary mean-state and terminal-state classifiers used only during training.

EVIDENCE: The otherwise identical deep-supervised 122-unit design achieved 85.77% accuracy at 1.145B MACs, matching the 123-unit design’s accuracy while using fewer MACs; this supports testing the adjacent lower-width boundary.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(242, 8)
        self.mean_aux = nn.Linear(121, 8)
        self.terminal_aux = nn.Linear(121, 8)
        self._aux_logits = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=-1
        )
        return self.classifier(features)
=======
        hidden, summary, count = state
        mean = summary / count.clamp_min(1.0)
        terminal = hidden[:, 0, :]
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2
>>>>>>> REPLACE
MECHANISM: Adjacent contextual-state width reduction

HYPOTHESIS: A 48/87 hierarchical GRU using all 23 qualified frames will retain at least 85% validation accuracy while reducing total inference MACs from 858,344,960 to approximately 845,791,515.

INTENDED_EDIT: Reconstruct the qualified 23-step hierarchy and shrink its contextual GRU, combined normalizations, classifier, auxiliary heads, and state tensors from 88 to 87 contextual units.

EVIDENCE: The otherwise identical 48/88 model qualified at 85.28% with 858,344,960 MACs, while both 22-step variants failed; an adjacent contextual-width reduction is the most direct remaining structural test without sacrificing temporal coverage.

<<<<<<< SEARCH
        self.context_gru = nn.GRU(48, 90, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(138)
        self.terminal_norm = nn.LayerNorm(138)
        self.classifier = nn.Linear(276, 8)
        self.mean_aux = nn.Linear(138, 8)
        self.terminal_aux = nn.Linear(138, 8)
=======
        self.context_gru = nn.GRU(48, 87, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(135)
        self.terminal_norm = nn.LayerNorm(135)
        self.classifier = nn.Linear(270, 8)
        self.mean_aux = nn.Linear(135, 8)
        self.terminal_aux = nn.Linear(135, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        context_hidden = torch.zeros(
            batch_size, 1, 90, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 90, device=device, dtype=dtype
        )
=======
        context_hidden = torch.zeros(
            batch_size, 1, 87, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 87, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE
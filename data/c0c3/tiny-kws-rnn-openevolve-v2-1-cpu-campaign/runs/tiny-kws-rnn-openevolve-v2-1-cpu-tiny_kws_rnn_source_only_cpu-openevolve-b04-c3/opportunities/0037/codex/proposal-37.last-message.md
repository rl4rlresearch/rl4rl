MECHANISM: Adjacent contextual-state width reduction

HYPOTHESIS: Reducing the contextual GRU from 90 to 89 units while preserving all 23 qualified frames will retain at least 85% validation accuracy and lower total inference MACs from 883,789,260 to approximately 871,010,875.

INTENDED_EDIT: Keep the qualified hierarchical architecture, schedule, and training procedure unchanged while shrinking the contextual state and corresponding normalization, readout, and auxiliary-head dimensions by one unit.

EVIDENCE: The current 48/90 hierarchy qualifies at 85.77% with 23 steps, whereas both tested 22-step schedules failed; prior adjacent-width experiments frequently qualified when sufficient margin remained, motivating structural width reduction without further temporal information loss.

<<<<<<< SEARCH
        self.context_gru = nn.GRU(48, 90, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(138)
        self.terminal_norm = nn.LayerNorm(138)
        self.classifier = nn.Linear(276, 8)
        self.mean_aux = nn.Linear(138, 8)
        self.terminal_aux = nn.Linear(138, 8)
=======
        self.context_gru = nn.GRU(48, 89, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(137)
        self.terminal_norm = nn.LayerNorm(137)
        self.classifier = nn.Linear(274, 8)
        self.mean_aux = nn.Linear(137, 8)
        self.terminal_aux = nn.Linear(137, 8)
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
            batch_size, 1, 89, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 89, device=device, dtype=dtype
        )
>>>>>>> REPLACE
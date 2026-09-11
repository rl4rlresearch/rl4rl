MECHANISM: Inference-head-tied dual-view supervision

HYPOTHESIS: A 119-unit GRU whose independently supervised mean and terminal heads are also used for inference will recover the failed 119-unit design’s 0.34-point accuracy deficit, reaching at least 85% while retaining approximately 1.053B inference MACs and reducing learned parameters.

INTENDED_EDIT: Reduce the recurrent width to 119, remove the redundant concatenated classifier, and average the separately supervised mean-state and terminal-state logits for the final prediction.

EVIDENCE: The prior 119-unit design missed qualification narrowly at 84.66%, while training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38%; tying those supervised views directly to inference targets the optimization gap without increasing inference MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
        self.mean_aux = nn.Linear(120, 8)
        self.terminal_aux = nn.Linear(120, 8)
        self._aux_logits = None
=======
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.mean_aux = nn.Linear(119, 8)
        self.terminal_aux = nn.Linear(119, 8)
        self._aux_logits = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits
=======
        mean_logits = self.mean_aux(mean)
        terminal_logits = self.terminal_aux(terminal)
        logits = 0.5 * (mean_logits + terminal_logits)
        if self.training:
            self._aux_logits = (mean_logits, terminal_logits)
        else:
            self._aux_logits = None
        return logits
>>>>>>> REPLACE
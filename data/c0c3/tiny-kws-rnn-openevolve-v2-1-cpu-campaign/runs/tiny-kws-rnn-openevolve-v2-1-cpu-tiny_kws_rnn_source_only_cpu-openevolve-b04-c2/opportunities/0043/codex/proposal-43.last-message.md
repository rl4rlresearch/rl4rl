MECHANISM: One-unit recurrent-width reduction with late-state-biased pooling

HYPOTHESIS: An 85-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 436,986,700.

INTENDED_EDIT: Reduce the qualified residual-pooled 20-step architecture from 86 to 85 hidden units while preserving batch size 16 and the established training procedure.

EVIDENCE: The 86-unit residual-pooled 20-step model achieved 86.38% at 446,333,120 MACs, improving the same-width mean-pooled model from 84.91%; this margin motivates the smallest unresolved structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
=======
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(85, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = 0.75 * mean_output + 0.25 * hidden[:, 0, :]
        return self.classifier(pooled)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE
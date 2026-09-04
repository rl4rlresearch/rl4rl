MECHANISM: Single temporal-summary coordinate pruning

HYPOTHESIS: Reducing the classifier input from 116 to 115 features by omitting one mean-summary coordinate will retain at least 85% validation accuracy while preserving all 21 recurrent steps and reducing validation inference by 6,520 MACs and eight parameters.

INTENDED_EDIT: Keep the verified GRU and frame schedule unchanged, but classify from 57 mean-summary features plus all 58 final-hidden features.

EVIDENCE: The 21-step width-58 model achieved 85.15%, whereas width-57 and several 20-step variants failed; this motivates a minimal classifier-only structural reduction that preserves recurrent capacity and every selected frame.

<<<<<<< SEARCH
        self.classifier = nn.Linear(116, 8)
=======
        self.classifier = nn.Linear(115, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))
=======
        return self.classifier(
            torch.cat((mean_output[:, :-1], hidden[:, 0, :]), dim=-1)
        )
>>>>>>> REPLACE
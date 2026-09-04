MECHANISM: Zero-MAC pooled temporal-summary compression

HYPOTHESIS: Pooling the final three mean-summary coordinates into one feature will retain at least 85% validation accuracy while reducing classifier input width from 114 to 113, saving 6,520 validation MACs and eight parameters.

INTENDED_EDIT: Replace the two discarded summary coordinates with a single elementwise pooled feature and shrink the classifier accordingly, preserving information from all 58 summary channels without learned matrix cost.

EVIDENCE: Dropping two summary coordinates retained 85.03% accuracy, but dropping a third fell to 83.93%; pooling the next coordinate with the already omitted coordinates directly tests whether preserving their aggregate information enables the same structural reduction.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)
=======
        self.classifier = nn.Linear(113, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
=======
        pooled_tail = mean_output[:, -3:].mean(dim=-1, keepdim=True)
        return self.classifier(
            torch.cat((mean_output[:, :-3], pooled_tail, hidden[:, 0, :]), dim=-1)
        )
>>>>>>> REPLACE
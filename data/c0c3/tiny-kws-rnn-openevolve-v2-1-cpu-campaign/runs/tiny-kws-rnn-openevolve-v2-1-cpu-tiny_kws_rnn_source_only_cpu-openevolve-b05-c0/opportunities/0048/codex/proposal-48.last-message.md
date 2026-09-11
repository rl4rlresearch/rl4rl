MECHANISM: Zero-MAC pooled temporal-summary compression

HYPOTHESIS: Compressing the final four mean-summary coordinates into one pooled feature will preserve at least 85% validation accuracy while reducing classifier width from 114 to 113, saving 6,520 validation MACs and eight parameters.

INTENDED_EDIT: Retain 54 individual mean-summary features, add their final four coordinates’ mean as one feature, retain all 58 final-hidden features, and shrink the classifier input accordingly.

EVIDENCE: Dropping two summary coordinates achieved 85.03%, while dropping a third fell to 83.93%; preserving aggregate information from the compressed coordinates directly tests whether the failed third-coordinate reduction was caused by information loss rather than classifier width.

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
        pooled_tail = mean_output[:, -4:].mean(dim=-1, keepdim=True)
        return self.classifier(
            torch.cat(
                (mean_output[:, :-4], pooled_tail, hidden[:, 0, :]), dim=-1
            )
        )
>>>>>>> REPLACE
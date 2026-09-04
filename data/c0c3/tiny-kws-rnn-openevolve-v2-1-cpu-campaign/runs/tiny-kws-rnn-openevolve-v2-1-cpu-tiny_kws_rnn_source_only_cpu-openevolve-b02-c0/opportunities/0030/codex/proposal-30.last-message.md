MECHANISM: Margin-sharpened four-step certified early exit

HYPOTHESIS: Reducing label smoothing from 0.03 to 0.02 will recover the one-example accuracy deficit of step-26 certified exit while increasing late-prefix margins, achieving at least 85% accuracy and fewer than 789,962,679 total inference MACs.

INTENDED_EDIT: Begin learned classification and conservative exit certification at recurrent step 26, and slightly sharpen the training targets without changing model size or inference architecture.

EVIDENCE: Step-26 certified exit reduced total MACs to 778,595,301 but missed the accuracy threshold by only one validation example at 84.91%; step 27 reached 85.03%, making a small training-only margin adjustment the most targeted verifiable follow-up after the auxiliary-loss implementations could not be verified.

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
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE
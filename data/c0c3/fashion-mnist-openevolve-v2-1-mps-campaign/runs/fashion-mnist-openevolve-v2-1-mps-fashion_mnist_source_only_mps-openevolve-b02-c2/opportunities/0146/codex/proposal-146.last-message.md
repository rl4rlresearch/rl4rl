MECHANISM: Stronger confidence-adaptive flip fusion

HYPOTHESIS: Increasing the margin-weight coefficient from 0.10 to 0.25 will exceed 9,322 correct predictions by resolving more flip-view disagreements in favor of the more decisive prediction while preserving the exact training path and reliable runtime.

INTENDED_EDIT: Strengthen only the evaluation-time probability-mixture weighting; leave architecture, optimization, augmentation, and example accounting unchanged.

EVIDENCE: Moving from equal probability fusion to coefficient 0.10 retained 9,320 correct while improving cross-entropy from 0.192262 to 0.192214, indicating that confidence weighting was directionally useful but too mild to change enough argmax decisions.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.25 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE
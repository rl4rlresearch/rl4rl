MECHANISM: Quadratic-vertex octic confidence calibration

HYPOTHESIS: An octic coefficient of −0.000000076055785 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194553794860838.

INTENDED_EDIT: Move the evaluation-time centered-octic coefficient from the continued probe to the quadratic optimum interpolated from the verified coefficients 0, −0.0000000477, and −0.0000000954.

EVIDENCE: The three octic coefficients produced cross-entropies 0.22194767456054687, 0.22194570770263672, and 0.22194553794860838 at identical accuracy; their diminishing improvement places the fitted minimum near −0.000000076055785.

<<<<<<< SEARCH
            - 0.0000000954 * (centered_octic - centered_octic.mean())
=======
            - 0.000000076055785 * (centered_octic - centered_octic.mean())
>>>>>>> REPLACE
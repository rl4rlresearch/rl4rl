MECHANISM: Endpoint-bisected margin-adaptive flip fusion

HYPOTHESIS: Raising the bounded adjustment to 0.0299609375 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.1962220642089844.

INTENDED_EDIT: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified setting and the 0.03 endpoint.

EVIDENCE: The verified 0.029921875 adjustment preserved 9,331 correct and achieved the best cross-entropy, 0.1962220642089844; every verified increase from 0.025 through 0.029921875 improved cross-entropy monotonically without changing correctness.

<<<<<<< SEARCH
                0.0296875
=======
                0.0299609375
>>>>>>> REPLACE
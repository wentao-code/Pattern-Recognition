# Accuracy Reports

Place reference prediction `.txt` files in this directory manually. Their
filename does not need a fixed format; when the final suffix before `.txt` is
a decimal such as `_0.86`, it represents a known accuracy of 86%.

Every new SVM or MLP result is compared by sample ID with all compatible
two-column prediction files here. The differing-label counts are printed and
stored in the run's metadata JSON. If there is no compatible reference, saving
stops. Scripts never modify files here, and difference counts do not calculate
the new model's test-set accuracy.

# Accuracy Reports

Place reference prediction `.txt` files in this directory manually. Their
filename does not need a fixed format; when the final suffix before `.txt` is
a decimal such as `_0.86`, it represents a known accuracy of 86%.

Both SVM and MLP training scripts read compatible two-column prediction files
from this directory because they use the same test samples. When sample IDs
match, the scripts print how many labels differ from the current predictions.
They do not write or modify files here and do not calculate test-set accuracy.

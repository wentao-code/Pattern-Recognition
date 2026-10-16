import numpy as np
from skimage.feature import hog


def to_hog_features(data, cell=3, orientations=8):
    return np.stack(
        [
            hog(
                row.reshape(19, 19),
                orientations=orientations,
                pixels_per_cell=(cell, cell),
                cells_per_block=(2, 2),
            )
            for row in data
        ]
    )

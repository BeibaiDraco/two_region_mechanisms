from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA


def pca_project(trajectory: np.ndarray, n_components: int = 2) -> Dict[str, np.ndarray]:
    """
    trajectory: [T, H] or [T*B, H]
    """
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(trajectory)
    return {
        "projection": proj,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }

from __future__ import annotations

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None


def _normalize_transition(matrix: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
    out = matrix / (matrix.sum(dim=0, keepdim=True) + 1e-8)
    out = out / (out.sum(dim=1, keepdim=True) + 1e-8)
    for _ in range(n_iter):
        out = out / (out.sum(dim=0, keepdim=True) + 1e-8)
        out = out / (out.sum(dim=1, keepdim=True) + 1e-8)
    return out


class CAARefiner:
    """Class-aware attention refinement adapted from CLIP-ES style affinity propagation."""

    def __init__(self, threshold: float = 0.4, n_iter: int = 2):
        self.threshold = threshold
        self.n_iter = n_iter

    def refine(self, cam_2d: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        cam_2d = cam_2d.float()
        affinity = affinity.float()

        h, w = cam_2d.shape
        flat_cam = cam_2d.reshape(-1, 1)

        trans_mat = _normalize_transition(affinity, n_iter=2)
        trans_mat = (trans_mat + trans_mat.t()) / 2.0
        trans_mat = torch.matmul(trans_mat, trans_mat)

        box_mask = self._build_box_mask(cam_2d, threshold=self.threshold).reshape(-1, 1)
        trans_mat = trans_mat * box_mask * box_mask.t()

        refined = flat_cam
        for _ in range(max(self.n_iter, 1)):
            refined = torch.matmul(trans_mat, refined)
            refined = refined / (refined.max() + 1e-8)

        return refined.reshape(h, w)

    def _build_box_mask(self, cam_2d: torch.Tensor, threshold: float) -> torch.Tensor:
        if cv2 is None:
            seed_mask = cam_2d > threshold
            if not seed_mask.any():
                return torch.ones_like(cam_2d)
            rows = seed_mask.any(dim=1)
            cols = seed_mask.any(dim=0)
            r_idx = torch.where(rows)[0]
            c_idx = torch.where(cols)[0]
            box = torch.zeros_like(cam_2d)
            box[int(r_idx.min()) : int(r_idx.max()) + 1, int(c_idx.min()) : int(c_idx.max()) + 1] = 1.0
            return box

        score = cam_2d.detach().float().cpu().numpy().astype(np.float32)
        max_val = float(score.max())
        if max_val <= 0:
            return torch.ones_like(cam_2d)

        score_u8 = np.expand_dims((score * 255.0).astype(np.uint8), axis=2)
        _, thr = cv2.threshold(
            score_u8,
            int(threshold * float(score_u8.max())),
            255,
            cv2.THRESH_BINARY,
        )
        contour_idx = 1 if cv2.__version__.split(".")[0] == "3" else 0
        contours = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[contour_idx]
        if len(contours) == 0:
            return torch.ones_like(cam_2d)

        box = torch.zeros_like(cam_2d)
        width = int(score.shape[1])
        height = int(score.shape[0])
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + bw, y + bh
            x1 = min(x1, width)
            y1 = min(y1, height)
            box[y0:y1, x0:x1] = 1.0

        return box

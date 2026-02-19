from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM for CNNs like ResNet.
    target_layer should be a nn.Module producing feature maps (e.g., model.layer4).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def close(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> Tuple[np.ndarray, float]:
        """
        Returns (heatmap [H,W] in 0..1, prob_of_class_idx)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        score = logits[:, class_idx].sum()
        score.backward()

        grads = self.gradients          # [B,C,H,W]
        acts = self.activations         # [B,C,H,W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        cam = (weights * acts).sum(dim=1)  # [B,H,W]
        cam = F.relu(cam)

        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.detach().cpu().numpy()
        prob = float(probs[0, class_idx].detach().cpu().item())
        return heatmap, prob

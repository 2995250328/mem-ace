import torch
import torch.nn as nn


class MCDropoutRegressor:
    """MC Dropout wrapper for a frozen ACE Regressor.

    Injects Dropout2d after every Conv2d in the head via forward hooks,
    then runs T stochastic forward passes to estimate per-pixel uncertainty.

    No separate network, no pretraining — uses the actual frozen head directly.

    Usage:
        mc = MCDropoutRegressor(regressor, dropout_p=0.1)
        samples = mc.coordinate_samples(features, T=10)  # (T, B, 3, Hf, Wf)
        var_map = samples.var(dim=0).mean(dim=1, keepdim=True)  # (B, 1, Hf, Wf)
    """

    def __init__(self, regressor: nn.Module, dropout_p: float = 0.1):
        self.regressor = regressor
        self.dropout = nn.Dropout2d(p=dropout_p)
        self._hooks: list = []

    # ── hook management ───────────────────────────────────────────────────────

    def _attach(self):
        """Register post-conv dropout hooks on every Conv2d in the head."""
        dropout = self.dropout
        dropout.train()

        def _hook(module, inp, out):
            return dropout(out)

        for module in self.regressor.heads.modules():
            if isinstance(module, nn.Conv2d):
                self._hooks.append(module.register_forward_hook(_hook))

    def _detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── MC inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def coordinate_samples(self, features: torch.Tensor, T: int = 10) -> torch.Tensor:
        """Run T stochastic forward passes through the frozen head.

        Args:
            features: (B, C, Hf, Wf) — frozen encoder features
            T:        number of MC samples

        Returns:
            (T, B, 3, Hf, Wf) scene coordinate samples
        """
        self._attach()
        try:
            samples = [
                self.regressor.get_scene_coordinates(features).float()
                for _ in range(T)
            ]
        finally:
            self._detach()
        return torch.stack(samples, dim=0)

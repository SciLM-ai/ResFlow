import torch
import torch.nn.functional as F

class FlowMatching:
    def __init__(self, model, drop_prob=0.1):
        self.model = model
        self.drop_prob = drop_prob

    def compute_loss(self, x1, cond, loss_weight=None):
        """Flow-matching velocity MSE.

        loss_weight: optional broadcastable tensor of per-voxel weights.
            Used for inpainting/outpainting training, where the known
            region is handed to the model as input and predicting it is
            trivial. With 1-voxel well masks that region is ~1% of the
            volume and including it is harmless, but a neighbour-context
            slab at overlap 24 covers 61% of a block, so an unweighted
            loss would be dominated by copying. The weighted mean uses
            the weight sum as its denominator so the loss stays on the
            same scale as the unweighted one.
            Defaults to None (unweighted), preserving prior behaviour.
        """
        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0],), device=x1.device)

        t_expand = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1 - t_expand) * x0 + t_expand * x1

        v_target = x1 - x0

        drop_mask = torch.rand(x1.shape[0], device=x1.device) < self.drop_prob
        v_pred = self.model(xt, t * 1000, cond, drop_mask=drop_mask)
        if loss_weight is None:
            return F.mse_loss(v_pred, v_target)
        w = loss_weight.to(v_pred.dtype)
        se = (v_pred - v_target) ** 2 * w
        return se.sum() / w.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, shape, device, cond=None, cfg_scale=3.0, n_steps=50):
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            t_emb = t * 1000  # scale for sinusoidal embedding

            if cond is not None and cfg_scale > 0:
                v_cond = self.model(x, t_emb, cond)
                v_uncond = self.model(x, t_emb)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.model(x, t_emb)

            x = x + v_pred * dt
        return x

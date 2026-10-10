"""ResFlow: 3D generative models for siliciclastic reservoirs.

    import resflow
    model = resflow.load_pretrained()          # the paper model, from huggingface.co/SciLM/ResFlow
    vols = model.generate('meander', n=8)
"""
from .pretrained import PretrainedResFlow, Well, load_pretrained

__all__ = ['load_pretrained', 'PretrainedResFlow', 'Well']

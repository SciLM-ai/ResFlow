"""Generate reservoirs with the pretrained ResFlow model (huggingface.co/SciLM/ResFlow).

    pip install git+https://github.com/SciLM-ai/ResFlow
    python examples/pretrained/quickstart.py

On a GPU this takes a few minutes in total; on a CPU it works but takes much longer
(lower ``n`` and the field size to try it out). Writes quickstart.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np

import resflow

model = resflow.load_pretrained()           # downloads the weights once (130 MB)

# 1. Volumes of one environment from its typical parameters.
#    Each volume is 64 x 64 x 32 cells (x, y, z; z = 0 is the base), 1 = sand, 0 = mud.
vols = model.generate('meander', n=8, seed=0)
print('meander volumes', vols.shape, 'sand fraction', round(float(vols.mean()), 3))

# 2. Choose parameters. Anything left out takes the environment's typical value;
#    parameters() lists each one as (typical, low, high) over the training data.
print(model.parameters('lobe'))
lobes = model.generate('lobe', n=4, ntg=0.35, width_cells=30, azimuth=45, seed=1)

# 3. Condition on a well: a vertical column of sand (1) and mud (0) at (x, y).
#    Here the column is taken from one of the volumes above; use -1 for unobserved cells.
column = vols[0, 32, 32]
ensemble = model.generate('meander', n=16, seed=2, wells=[resflow.Well(x=32, y=32, facies=column)])
assert (ensemble[:, 32, 32] == column).all()          # every realisation honours the well exactly
p_sand = ensemble.mean(axis=0)                         # per-cell probability of sand

# 4. A whole field in one pass, here with lobes that shrink along x.
#    Any parameter can be a scalar or a (X, Y) map; wells work on fields too.
X, Y = 512, 512
width = np.linspace(60, 30, X)[:, None].repeat(Y, axis=1)
field = model.generate_field('lobe', shape=(X, Y, 32), width_cells=width, ntg=0.5, seed=3)

# 5. Look at horizontal sections (z = 16).
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
panels = [(vols[0], 'meander volume'), (lobes[0], 'lobe volume, ntg 0.35, azimuth 45'),
          (p_sand, 'P(sand), 16 runs honouring the well'), (field, 'lobe field, lobes shrink along x')]
for a, (v, title) in zip(ax, panels):
    im = a.imshow(v[:, :, 16].T, origin='lower', cmap='YlOrBr' if v is not p_sand else 'viridis',
                  vmin=0, vmax=1, interpolation='nearest')
    a.set_title(title, fontsize=10)
    a.set_xlabel('x (cells)')
ax[0].set_ylabel('y (cells)')
ax[2].plot([32], [32], 'r+', ms=12)
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout()
fig.savefig('quickstart.pdf')
print('wrote quickstart.pdf')

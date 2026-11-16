# PhysICL: Physics-Prior In-Context Learning for Few-Shot Bearing Fault Diagnosis on Unseen Machines

Code for the paper

> M. Kurnaz and C. Alagöz, "PhysICL: Physics-Prior In-Context Learning for Few-Shot Bearing Fault Diagnosis on Unseen
> Machines," submitted to *IEEE Transactions on Instrumentation and Measurement*, 2026.

PhysICL fine-tunes the in-context time-series classifier [TimEE](https://huggingface.co/liamsbhoo/timee) **only on
episodes from a randomized physics simulator of rotating machines**, with no real data, and queries it with the raw
vibration window and its Hilbert envelope. On a new machine the model receives a few labelled windows per health state
and classifies the remaining windows in one forward pass, without any weight update.

```
randomized simulator (machine → part → condition)  ──►  episodic fine-tuning of TimEE  ──►  frozen model
                                                                                              │
new machine: k labelled windows per class + unlabelled queries ──► raw view + envelope view ──┴──► p̄ = ½(p_raw + p_env)
```

## Repository layout

| Path | Contents |
| --- | --- |
| `pilot2/simulator.py` | Randomized physics prior: bearing kinematics (BPFO/BPFI/BSF), impact trains with slip, load-zone modulation, structural modes, sensor band limit, shaft/gear/line components; machine, part and condition levels; 8 health states |
| `pilot2/generic_prior.py` | Difficulty-matched **non-physical control prior** (sines, AR processes, spikes, AM, random walks, chirps) |
| `pilot2/finetune.py` | Episodic fine-tuning of TimEE on either prior |
| `pilot2/views.py` | Raw and envelope (500-Hz high-pass + Hilbert) views at inference |
| `pilot2/evaluate.py`, `pilot2/analysis.py` | Evaluation of fine-tuned models and the preregistered statistical analysis (paired Wilcoxon, Holm, non-inferiority) |
| `pilot2/plan.py` | The three paper stages A (held-out datasets, training seeds), B (ablations), C (simulator variant, replication) |
| `pilot2/fidelity.py` | Simulator–measurement coverage analysis (Section V-C of the paper) |
| `pilot2/ONKAYIT*.md` | Preregistrations, written and dated before each confirmatory run (in Turkish, unchanged) |
| `pilot1/` | Data loaders (CWRU, SEU, Paderborn, JNU, HIT-SM), leakage-free episode sampler, baselines (TimEE, MantisV2 + LR, envelope spectrum, WDCNN, ProtoNet) |
| `run_pilot1.py`, `run_pilot2.py`, `pu_sanity.py` | Entry points (settings block at the top of each file, or `key=value` arguments) |
| `figures/` | Scripts that produce the figures of the paper from the result files |

Code comments and console messages are in Turkish; identifiers and this README are in English where it matters.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Paderborn archives are `.rar` files and need 7-Zip (Windows) or `unrar` (Linux). TensorFlow is only needed for the
WDCNN and ProtoNet baselines; on Windows import PyTorch before TensorFlow (the entry points already do this).

## Data

All six datasets are public and are downloaded automatically into `data/` on first use:

| Dataset | Source used by the loader |
| --- | --- |
| CWRU, SEU | `github.com/cathysiyu/Mechanical-datasets` |
| Paderborn (PU) | Zenodo record of the KAt bearing data |
| JNU | `github.com/ClarkGableWang/JNU-Bearing-Dataset` |
| HIT-MFS, HIT-SB | `github.com/hitwzc/Bearing-datasets` |

All signals are resampled to 12 kHz and cut into non-overlapping 1024-sample windows. Support and query windows never
share a recording; the grouping that separates them (operating condition, bearing or defect) is fixed per dataset in
`pilot1/data.py` (`SPLIT_MODE`).

## Reproducing the paper

To check the installation, `python run_pilot2.py adim=ABC --test` and `python run_pilot1.py --synthetic --random-init-fm`
run the whole pipeline in a few minutes on synthetic data with random weights.

```bash
# Baselines on the development datasets (Table III, external baselines)
python run_pilot1.py --data-root data --datasets cwru,seu,pu --methods timee_raw,mantisv2_lr,env_proto

# Fine-tune PhysICL (physics prior) and the control prior, 2000 steps each
python run_pilot2.py adim=ince_ayar onsel=fizik etiket=fizik_v1
python run_pilot2.py adim=ince_ayar onsel=genel etiket=genel_v1

# Preregistered confirmation on new development episodes (seed 1)
python run_pilot2.py adim=v1_dogrulama

# Paper stages: A = held-out datasets + 3 training seeds, B = ablations, C = simulator v1 + replication (seed 2)
python run_pilot2.py adim=ABC

# Simulator–measurement coverage analysis
python run_pilot2.py adim=uyum

# Same-bearing vs. different-bearing check on Paderborn (Fig. 1)
python pu_sanity.py
```

Results are written to `results/` and `results_pilot2/` (`A_sonuc.md`, `B_sonuc.md`, `C_sonuc.md`, per-episode CSV files).
Naming: `fizik_*` is the physics prior (PhysICL), `genel_*` the control prior, `hazir` the untouched TimEE; the view
suffix `|ham+zarf` is raw + envelope.

Figures:

```bash
python figures/make_figures.py results_pilot2 figures/out
PYTHONPATH=. python figures/method_figure.py figures/out
PYTHONPATH=. python figures/fidelity_figure.py results_pilot2/uyum figures/out
```

## Using the simulator on its own

```python
import numpy as np
from pilot2.simulator import sample_machine, sample_unit, render, sample_episode

rng = np.random.default_rng(0)
m = sample_machine(rng)                 # a virtual machine: geometry, modes, band limit, background
u = sample_unit(m, "OR", rng)           # one part of that machine with an outer-race defect
x = render(m, u, cond=0, n=12_000, rng=rng)   # 1 s of vibration at 12 kHz

episode = sample_episode(rng)           # a TimEE-style training episode (support, then queries)
```

Health states: `N, OR, IR, B, OR+IR, UNB, MIS, LOOSE`.

## Hardware

On the authors' single consumer GPU, stages A and B took about 1.5 hours each and stage C about 1 hour. Episodes are
generated on CPU workers (about 50 ms per episode on one core). TimEE and MantisV2 evaluation needs a GPU.

## Citation

```bibtex
@article{kurnaz2026physicl,
  author  = {Kurnaz, Mehmet and Alag{\"o}z, Celal},
  title   = {{PhysICL}: Physics-Prior In-Context Learning for Few-Shot Bearing Fault Diagnosis on Unseen Machines},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  note    = {Submitted},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. The datasets keep the licenses of their original providers; TimEE and MantisV2 weights are
downloaded from their official sources under their own licenses.

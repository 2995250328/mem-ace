# Step 06 - Compressor Geometry Contract

Status: designing

## Purpose

This step tracks compressor-side ideas from `silu/修改.md` that are useful but
should not be mixed into the first GeoMatch Fusion change.

This step is still a near-term ACE-G architecture item. It only concerns the
geometry encodings consumed by the current compressor/fusion modules. It is not
the same as the long-term reference-frame memory contract or full normalized
coordinate target training.

The compressor already has meaningful geometry:

- FPS latent coordinates.
- Query PE from `latent_coords - scene_center`.
- Key feature from one selected memory layer.
- Value feature from all-layer concatenation.
- Global attention with learned distance bias.
- Optional scale token global shift.

The problem is not that the compressor has no geometry. The problem is that
some geometry choices are still hard to interpret or scene-scale dependent.

## Problems

### 1. Key Layer Is Explicit But Not Yet Understood

Current compatibility behavior:

- multi-layer memory defaults to `key_slice_idx=2`
- with current memory layers this corresponds to layer 12

Hypothesis:

- layer 12 / layer 18 may be better key layers than slice 0 or final
- final may be semantically strong but less locally discriminative
- slice 0 may be too raw, possibly before MapAnything information sharing

Recommended ablation order:

```text
A0: layer 12 / key_slice_idx=2
A1: layer 18 / key_slice_idx=3
A2: layer 6  / key_slice_idx=1
A3: final    / key_slice_idx=4
A4: learned scalar mix for key only
```

Keep value as all-layer concatenation during these ablations.

For learned scalar mix:

```python
key_feat = sum_l softmax(w_l) * feat_l
value = concat(all_layers)
```

Rationale:

- Key remains one-layer dimensionality.
- Parameter and compute growth are limited.
- Value still preserves full memory content.

### 2. PE And Distance Bias Need A Scene-Scale Contract

Current behavior:

```text
coords = points - scene_center
PE(coords)
distance bias uses raw coordinate distances
```

Problem:

- Indoor and outdoor scenes can have very different coordinate ranges.
- Random Fourier PE frequencies mean different physical scales across scenes.
- Distance bias magnitude can become scene-scale dependent.

Near-term contract:

```python
coords_norm = (coords - scene_center) / scene_scale
```

Potential `scene_scale` choices:

- bbox radius
- point standard deviation
- percentile radius

The same scene-scale policy should be used for:

- compressor PE
- fusion PE
- global distance bias

Do not confuse this with full normalized-coordinate target training. This step
only normalizes geometry encodings, not the ACE coordinate target.

Boundary with long-term normalization:

- Here, the ACE head still predicts the current training coordinate target.
- `scene_scale` only changes inputs to PE / geometry bias.
- A future normalized target path would instead change the model output space
  and would require explicit recovery metadata before evaluation.
- If reference-frame memory is later introduced, the full chain becomes
  `world -> reference -> normalized`; this step does not implement that chain.

Metadata that must be logged when this becomes code:

- `geometry_scene_scale_policy`
- `geometry_scene_scale_value`
- coordinate frame used for compressor/fusion PE
- whether distance bias uses raw or scale-normalized distances
- whether fusion PE and compressor PE share the same scale policy

### 3. Fourier PE Ablations Should Wait For Scene-Scale Contract

Candidate PE variants:

- current random Fourier baseline
- fixed multi-scale Fourier
- fixed Fourier + raw xyz
- fixed Fourier + raw xyz + radius
- shared frequency basis between compressor and fusion

These are lower priority than scene-scale normalization, because without a
shared scale policy the PE ablation is hard to interpret.

### 4. Global Distance Bias Is Currently Too Narrow

Current simplified form:

```python
geo_bias = MLP(log(dist_sq))
attn_logits = content_logits + geo_bias
```

Potential future basis:

```python
dist_norm = sqrt(dist_sq) / scene_scale
geo_feat = [
    log1p(dist_norm),
    dist_norm,
    exp(-dist_norm^2 / sigma1^2),
    exp(-dist_norm^2 / sigma2^2),
    exp(-dist_norm^2 / sigma3^2),
]
geo_bias = MLP(geo_feat)
attn = content_logits + alpha_head * geo_bias
```

Add a learnable `alpha_head` gate initialized small so training starts close to
content-only attention.

Direction features:

```python
direction = (memory_point - latent_coord) / distance
```

This is plausible but should come after normalized distance basis, because it
adds more complexity and more ways to destabilize attention.

## Priority

Order within this step:

1. Log current compressor geometry stats and key metadata.
2. Run key-layer ablations.
3. Define and log scene-scale policy.
4. Apply scene-scale to PE/distance bias as a controlled ablation.
5. Only then test PE variants or richer distance-bias basis.

## Non-Goals

- Do not change memory extraction.
- Do not change ACE head.
- Do not implement DSD/deformable token movement here.
- Do not switch to full normalized-coordinate target training.
- Do not convert memory points into reference-frame coordinates here.
- Do not introduce multi-scene batching or sub-memory routing here.
- Do not combine key-layer ablation with GeoMatch Fusion v1 in the same first
  experiment.

## Success Criteria

- Key-layer ablations are interpretable from checkpoint config and logs.
- Scene-scale policy is recorded in checkpoint metadata.
- PE/distance-bias ablations use the same scale policy in compressor and fusion.
- Any gain can be attributed to one changed variable.

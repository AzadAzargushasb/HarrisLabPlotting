# Node-role deep dive

When you supply per-node graph metrics — specifically the **participation
coefficient (P)** and the **within-module Z-score** — `hlplot modular`
classifies every ROI into one of seven cartographic roles from
Guimerà & Nunes Amaral (2005)[^1] and draws each node with a colored border
encoding its role. The fill stays module-colored; the border is the role.

This dual encoding lets a single static figure communicate both the modular
structure and which nodes are integrating it.

[^1]: Guimerà R, Nunes Amaral LA. *Functional cartography of complex
    metabolic networks.* **Nature** 433, 895–900 (2005).
    [doi:10.1038/nature03288](https://doi.org/10.1038/nature03288)

## The two-cut classification

A node is a **hub** if its within-module Z-score is high (it has many
connections inside its own module). A node is a **connector** if its
participation coefficient is high (it spreads its connections across many
modules). Putting both axes on a 2-D plot gives the seven-region
"functional cartography":

```
                    P (participation coefficient)
                    0.0    0.30   0.62   0.75  0.80   1.0
                    │       │      │      │     │      │
        Z >= 2.5    │  R5   │      R6     │     R7     │   ← Hubs
              ──────┼───────┴──────┬──────┴─────┬──────┤
        Z <  2.5    │R1│   R2     │     R3      │  R4  │   ← Non-hubs
                    │  │          │             │      │
                    0  0.05      0.62          0.80    1.0
                              P (participation coefficient)
```

The hub cut is at **Z = 2.5**; the connector cuts depend on whether you're
above or below that line.

## The seven roles

| Role | Hub? | P range | Border color | What it means |
| --- | --- | --- | --- | --- |
| **R1 Ultra-peripheral** | non-hub | P ≤ 0.05 | `#FFFFFF` white | Connects only inside its own module |
| **R2 Peripheral** | non-hub | 0.05 < P ≤ 0.62 | `#CCCCCC` light gray | Mostly intra-module |
| **R3 Non-hub connector** | non-hub | 0.62 < P ≤ 0.80 | `#00CED1` turquoise | Bridges modules without being a hub |
| **R4 Non-hub kinless** | non-hub | P > 0.80 | `#FF1493` deep pink | Spread very thin across modules |
| **R5 Provincial hub** | hub | P ≤ 0.30 | `#FFFF00` yellow | Hub for its own module only |
| **R6 Connector hub** | hub | 0.30 < P ≤ 0.75 | `#000000` black | Hub that links its module to others |
| **R7 Kinless hub** | hub | P > 0.75 | `#FF00FF` magenta | Hub with no clear modular home |

NaN/inf inputs get a fallback `"Unclassified"` role with a `#808080` border.

## Producing a role-classified plot

You need a `--node-metrics` CSV with at minimum these two columns:

```text
roi_index,roi_name,participation_coefficient,within_module_zscore
1,rACC_L,0.42,1.18
2,rACC_R,0.51,2.71
...
```

Then:

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --node-size both \
  --output role_classified.html
```

`--node-size both` sizes nodes by `pc × zscore`, so the visual weight aligns
with the role classification.

```{interactive-plot}
:image: images/cli_tutorial/12a_q_z.png
:caption: Modular network with seven-role border coloring. Yellow rings = provincial hubs; black rings = connector hubs; magenta = kinless hubs.
:height: 540
```

## Reading a role-classified figure

Three quick patterns to look for:

1. **Yellow rings clustered inside one module** — that module is internally
   well-organized (provincial hubs).
2. **Black rings on the boundary between modules** — these are the bridges
   that hold the network together. Lesion any of them and you fragment the
   modular structure.
3. **Magenta or deep-pink rings** — relatively rare in healthy data; these
   nodes don't sit cleanly inside any module. Often interesting case studies.

## In Python

The classifier is a single function call:

```python
from HarrisLabPlotting import classify_node_role

role, hex_color = classify_node_role(z_score=2.71, pc=0.51)
# → ('Connector hub', '#000000')
```

Use it directly when you want to drive your own visualization, or let
`create_enhanced_modularity_visualization` apply it to every ROI for you.

## See also

- The standard module-coloring story: [Modularity tutorial](modularity.md)
- Filter to inter-module edges only:
  see `--visualization-type inter` in [CLI walkthrough §12](cli_walkthrough.md#12-modularity-visualization)
- Combine roles with statistical significance:
  [P-value plotting §7](pvalue_plotting.md)

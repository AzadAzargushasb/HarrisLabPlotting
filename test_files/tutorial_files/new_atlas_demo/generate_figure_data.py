"""
Generate all data for the "Figure Creation" tutorial (new human + monkey atlases).

Everything this writes is **synthetic** (the connectivity matrices, modules,
node sizes and metrics are fabricated with fixed random seeds). The ROI
*coordinates* and *names*, however, are real: they come from the actual atlas
volumes + label tables on disk.

Inputs (already on disk, under ``../parcellation and meshes/``):
  Human  : MMP_in_MNI_corr.nii.gz   (360-region HCP-MMP1, MNI152 ICBM2009a)
           roi_names.csv            (regionID -> regionName, e.g. V1_L)
           HCPMMP1_..._hd_0.obj     (surface mesh)
  Monkey : MacBNA__LR_304.nii.gz    (304-region Macaque Brainnetome, native space)
           Nomenclature_MBNA_304.xlsx (152 finer regions x L/R label IDs)
           monkey_brain_mesh_MacBNA.obj

Outputs (written under ``human/`` and ``monkey/``):
  human/hcpmmp1_labels.txt          (LUT: regionID<TAB>regionName, 360 rows)
  human/hcpmmp1_coords.csv          (360 ROI COGs)
  human/hcpmmp1_modules.csv         (roi_index,module ; 5 spatial modules)
  human/hcpmmp1_modular_network.csv (360x360 synthetic matrix, 50 edges)
  monkey/macbna_labels.txt          (LUT: label<TAB>name, 304 rows)
  monkey/macbna_coords.csv          (304 ROI COGs)
  monkey/coords_28.csv              (28-ROI spread subset for the legend demos)
  monkey/sizes_from_pc.csv          (per-node sizes pre-scaled from PC)
  monkey/metrics.csv                (per-node participation_coef + z-score)

Run with the project's env, e.g.:
  /home/aazarg/.conda/envs/pre_env/bin/python generate_figure_data.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from HarrisLabPlotting import coordinate_function, check_coords_in_mesh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                       # .../new_atlas_demo
PARC = HERE.parent / "parcellation and meshes"
HUMAN = HERE / "human"
MONKEY = HERE / "monkey"
HUMAN.mkdir(parents=True, exist_ok=True)
MONKEY.mkdir(parents=True, exist_ok=True)

HUMAN_NII = PARC / "HCPMMP1_on_MNI152_ICBM2009a_nlin_hd.nii" / "MMP_in_MNI_corr.nii.gz"
HUMAN_ROI_CSV = PARC / "HCPMMP1_on_MNI152_ICBM2009a_nlin_hd.nii" / "roi_names.csv"
HUMAN_MESH = PARC / "HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj"

MONKEY_NII = PARC / "MacBNA_parcellation_for_Publish(1)" / "vol" / "MacBNA__LR_304.nii.gz"
MONKEY_XLSX = PARC / "MacBNA_parcellation_for_Publish(1)" / "Nomenclature_MBNA_304.xlsx"
MONKEY_MESH = PARC / "monkey_brain_mesh_MacBNA.obj"

EXAMPLE_EDGE = HERE.parent / "node_edge_28" / "connectivity_28.edge"


# ---------------------------------------------------------------------------
# LUTs (index<TAB>name) from the real atlas label tables
# ---------------------------------------------------------------------------
def build_human_lut() -> Path:
    """360-region HCP-MMP1 LUT: regionID -> regionName (e.g. 1 -> V1_L)."""
    df = pd.read_csv(HUMAN_ROI_CSV)
    lut = HUMAN / "hcpmmp1_labels.txt"
    lines = [f"{int(r.regionID)}\t{r.regionName}" for r in df.itertuples()]
    lut.write_text("\n".join(lines) + "\n")
    print(f"[human] wrote LUT with {len(lines)} regions -> {lut.name}")
    return lut


def build_monkey_lut() -> Path:
    """
    304-region MacBNA LUT from Nomenclature_MBNA_304.xlsx.

    152 finer regions, each with a `label (Left)` (1-152) and `label (right)`
    (153-304) ID. Names are `<abbr>_left` / `<abbr>_right` (e.g. FP.d_left).
    """
    df = pd.read_excel(MONKEY_XLSX, sheet_name=0)
    abbr = df["abbr..1"].astype(str).str.strip()
    rows = {}
    for i, row in df.iterrows():
        name = abbr.iloc[i]
        if name in ("", "nan", "None"):
            name = f"ROI{i + 1:03d}"
        rows[int(row["label (Left)"])] = f"{name}_left"
        rows[int(row["label (right)"])] = f"{name}_right"
    lut = MONKEY / "macbna_labels.txt"
    lines = [f"{idx}\t{rows[idx]}" for idx in sorted(rows)]
    lut.write_text("\n".join(lines) + "\n")
    print(f"[monkey] wrote LUT with {len(lines)} regions -> {lut.name}")
    return lut


# ---------------------------------------------------------------------------
# Coordinates (real COGs from the atlas volume via the package function)
# ---------------------------------------------------------------------------
def gen_coords(nii: Path, lut: Path, name: str, dest: Path) -> pd.DataFrame:
    """Run coordinate_function into a temp dir, copy the comma CSV to ``dest``."""
    tmp = Path(tempfile.mkdtemp())
    coordinate_function(str(nii), str(lut), name_of_file=name,
                        save_directory=str(tmp), round_labels=True)
    src = tmp / f"{name}_comma.csv"
    shutil.copy(src, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.read_csv(dest)
    n_nan = int(df["cog_x"].isna().sum())
    print(f"[{name}] coords: {len(df)} ROIs, {n_nan} NaN COGs -> {dest.name}")
    return df


# ---------------------------------------------------------------------------
# Synthetic networks
# ---------------------------------------------------------------------------
def gen_human_modular(coords: pd.DataFrame, n_modules: int = 5,
                      per_module: int = 6, n_edges: int = 50,
                      seed: int = 42) -> None:
    """
    5 spatially-compact modules (KMeans on COGs) + a 50-edge synthetic matrix.

    Edges are placed mostly within modules (≈40 intra + ≈10 inter) among a
    compact subset of ``per_module`` nodes nearest each module centroid, so the
    rendered network shows five tight, anatomically-coherent clusters.
    """
    xyz = coords[["cog_x", "cog_y", "cog_z"]].to_numpy(dtype=float)
    n = len(coords)
    km = KMeans(n_clusters=n_modules, random_state=0, n_init=10).fit(xyz)
    modules = km.labels_ + 1  # 1..n_modules

    pd.DataFrame({"roi_index": np.arange(n), "module": modules}).to_csv(
        HUMAN / "hcpmmp1_modules.csv", index=False)

    # Compact subset: the `per_module` ROIs nearest each module centroid.
    subset = []
    for k in range(n_modules):
        members = np.where(km.labels_ == k)[0]
        d = np.linalg.norm(xyz[members] - km.cluster_centers_[k], axis=1)
        subset.extend(members[np.argsort(d)[:per_module]].tolist())
    subset = np.array(sorted(subset))

    intra, inter = [], []
    for ai in range(len(subset)):
        for bi in range(ai + 1, len(subset)):
            a, b = int(subset[ai]), int(subset[bi])
            (intra if modules[a] == modules[b] else inter).append((a, b))

    rng = np.random.default_rng(seed)
    rng.shuffle(intra)
    rng.shuffle(inter)
    n_inter = min(10, len(inter), n_edges)
    n_intra = n_edges - n_inter
    chosen = intra[:n_intra] + inter[:n_inter]

    mat = np.zeros((n, n))
    for a, b in chosen:
        w = float(rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0]))
        mat[a, b] = mat[b, a] = w
    pd.DataFrame(mat).to_csv(HUMAN / "hcpmmp1_modular_network.csv",
                             index=False, header=False)

    connected = sorted({i for e in chosen for i in e})
    print(f"[human] modular net: {len(chosen)} edges, {n_modules} modules, "
          f"{len(connected)} connected nodes -> hcpmmp1_modular_network.csv")


def gen_monkey_network(coords: pd.DataFrame, n_nodes: int = 28,
                       seed: int = 0) -> None:
    """
    28 spatially-spread MacBNA ROIs + PC-derived node sizes/metrics.

    Reuses the bundled 28-node example connectivity (``connectivity_28.edge``);
    we only fabricate the per-node sizes/metrics (as in the legend tutorial §1-2)
    and choose which 28 real monkey ROIs to hang the example topology on.
    """
    xyz = coords[["cog_x", "cog_y", "cog_z"]].to_numpy(dtype=float)
    km = KMeans(n_clusters=n_nodes, random_state=0, n_init=10).fit(xyz)
    chosen = []
    for k in range(n_nodes):
        members = np.where(km.labels_ == k)[0]
        d = np.linalg.norm(xyz[members] - km.cluster_centers_[k], axis=1)
        chosen.append(int(members[np.argmin(d)]))
    chosen = sorted(set(chosen))
    coords28 = coords.iloc[chosen].reset_index(drop=True)
    coords28.to_csv(MONKEY / "coords_28.csv", index=False)

    rng = np.random.default_rng(seed)
    pc = rng.uniform(0.05, 0.92, size=n_nodes)
    pd.DataFrame({
        "roi_name": coords28["roi_name"].values,
        "participation_coef": pc,
        "within_module_zscore": rng.normal(0, 1.5, size=n_nodes),
    }).to_csv(MONKEY / "metrics.csv", index=False)
    pd.DataFrame({"size": 5 + pc * 25}).to_csv(MONKEY / "sizes_from_pc.csv",
                                               index=False)
    print(f"[monkey] picked {len(coords28)} spread ROIs -> coords_28.csv; "
          f"wrote sizes_from_pc.csv + metrics.csv (PC range "
          f"{pc.min():.2f}-{pc.max():.2f})")


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Building LUTs ...")
    human_lut = build_human_lut()
    monkey_lut = build_monkey_lut()

    print("\nGenerating coordinates (real COGs) ...")
    human_coords = gen_coords(HUMAN_NII, human_lut, "hcpmmp1",
                              HUMAN / "hcpmmp1_coords.csv")
    monkey_coords = gen_coords(MONKEY_NII, monkey_lut, "macbna",
                               MONKEY / "macbna_coords.csv")

    print("\nGenerating synthetic networks ...")
    gen_human_modular(human_coords)
    gen_monkey_network(monkey_coords)

    print("\nAlignment sanity checks (COGs vs mesh) ...")
    h = check_coords_in_mesh(HUMAN / "hcpmmp1_coords.csv", str(HUMAN_MESH))
    m = check_coords_in_mesh(MONKEY / "macbna_coords.csv", str(MONKEY_MESH))
    print(f"  human : {h['verdict']}  (inside {h['n_inside']}/"
          f"{h['n_inside'] + h['n_outside']}, midline {h['midline_fraction']:.0%})")
    print(f"  monkey: {m['verdict']}  (inside {m['n_inside']}/"
          f"{m['n_inside'] + m['n_outside']}, midline {m['midline_fraction']:.0%})")
    print("=" * 70)
    print("Done. Data written under human/ and monkey/.")


if __name__ == "__main__":
    main()

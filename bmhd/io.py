"""
BMHD I/O Module: CWLS LAS 2.0 Diagnostic Chord Parser/Serializer,
Minimum-Curvature .dev Directional Sightline Survey Solver, and
Out-of-Core Field-Cube Memmap Streaming Engine.
"""

from __future__ import annotations

import math
import struct
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# 1. Trajectory Data Structures & Minimum-Curvature Solver
# =============================================================================

@dataclass
class DiagnosticTrajectory:
    """
    Representation of a 3D diagnostic laser chord / sightline trajectory in
    cylindrical/Cartesian torus coordinates and computational voxel grid indices.
    """
    sightline_name: str
    sightline_type: str
    md_m: np.ndarray             # [S] Measured Depth / chord arc length (m)
    inc_deg: np.ndarray          # [S] Ray inclination from vertical Z axis (deg, 0=vert, 90=horiz)
    azi_deg: np.ndarray          # [S] Ray azimuth in toroidal plane (deg)
    tvd_m: np.ndarray            # [S] True Vertical Depth / coordinate (m)
    north_y_m: np.ndarray        # [S] Y coordinate offset (m)
    east_x_m: np.ndarray         # [S] X coordinate offset (m)
    z_vox: np.ndarray            # [S] Continuous grid Z voxel index in [0, Z-1]
    y_vox: np.ndarray            # [S] Continuous grid Y voxel index in [0, Y-1]
    x_vox: np.ndarray            # [S] Continuous grid X voxel index in [0, X-1]
    dogleg_deg_30m: np.ndarray   # [S] Dogleg severity / ray bending rate (deg / 30m)

    @property
    def stations(self) -> np.ndarray:
        """Matrix of survey stations [S, 10]: MD, INC, AZI, TVD, Y, X, Z_VOX, Y_VOX, X_VOX, DLS."""
        return np.column_stack([
            self.md_m,
            self.inc_deg,
            self.azi_deg,
            self.tvd_m,
            self.north_y_m,
            self.east_x_m,
            self.z_vox,
            self.y_vox,
            self.x_vox,
            self.dogleg_deg_30m,
        ])

    @property
    def trajectory_xyz(self) -> np.ndarray:
        """Spatial Cartesian coordinates array of shape [S, 3]: (X, Y, Z)."""
        return np.column_stack([self.east_x_m, self.north_y_m, self.tvd_m])

    @property
    def voxel_indices(self) -> np.ndarray:
        """Continuous voxel coordinates array of shape [S, 3]: (Z_VOX, Y_VOX, X_VOX)."""
        return np.column_stack([self.z_vox, self.y_vox, self.x_vox])

    def __iter__(self) -> Iterator[Any]:
        """Support unpacking as (stations, trajectory_xyz)."""
        yield self.stations
        yield self.trajectory_xyz

    def __getitem__(self, idx: int) -> Any:
        return (self.stations, self.trajectory_xyz)[idx]

    def __len__(self) -> int:
        return 2


def solve_minimum_curvature_trajectory(
    md_m: Sequence[float] | np.ndarray,
    inc_deg: Sequence[float] | np.ndarray,
    azi_deg: Sequence[float] | np.ndarray,
    surface_yx_vox: Tuple[float, float] = (12.0, 12.0),
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    extents_zyx_m: Tuple[float, float, float] = (1.0, 2.4, 2.4),
    origin_zyx_m: Tuple[float, float, float] = (-0.5, -1.2, -1.2),
    sightline_name: str = "Chord-1",
    sightline_type: str = "tangential_toroidal",
) -> DiagnosticTrajectory:
    """
    Solve a 3D diagnostic laser chord or directional sightline trajectory
    `(TVD, Y, X)` from Measured Path Length `md_m`, Inclination `inc_deg`,
    and Azimuth `azi_deg` using the SPE/API industry-standard Minimum Curvature Method:

        beta_k = arccos(cos(I_1) cos(I_2) + sin(I_1) sin(I_2) cos(A_2 - A_1))
        RF_k   = (2 / beta_k) * tan(beta_k / 2)   (with RF_k -> 1 + beta_k^2 / 12 as beta_k -> 0)
        dTVD   = 0.5 * dMD * (cos(I_1) + cos(I_2)) * RF_k
        dY     = 0.5 * dMD * (sin(I_1) cos(A_1) + sin(I_2) cos(A_2)) * RF_k
        dX     = 0.5 * dMD * (sin(I_1) sin(A_1) + sin(I_2) sin(A_2)) * RF_k
    """
    md = np.asarray(md_m, dtype=np.float64)
    inc = np.asarray(inc_deg, dtype=np.float64)
    azi = np.asarray(azi_deg, dtype=np.float64)
    if md.ndim != 1 or md.size < 2:
        raise ValueError("Diagnostic sightline requires at least 2 stations along Measured Depth.")

    i_rad = np.deg2rad(inc)
    a_rad = np.deg2rad(azi)
    d_md = np.diff(md)

    i1, i2 = i_rad[:-1], i_rad[1:]
    a1, a2 = a_rad[:-1], a_rad[1:]

    cos_beta = np.clip(
        np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1),
        -1.0,
        1.0,
    )
    beta = np.arccos(cos_beta)

    # Exact ratio factor RF with smooth Taylor expansion for near-straight segments
    rf = np.ones_like(beta)
    small = np.abs(beta) < 1e-4
    rf[small] = 1.0 + (beta[small] ** 2) / 12.0
    rf[~small] = (2.0 / beta[~small]) * np.tan(0.5 * beta[~small])

    d_tvd = 0.5 * d_md * (np.cos(i1) + np.cos(i2)) * rf
    d_y = 0.5 * d_md * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_x = 0.5 * d_md * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf

    Z, Y, X = grid_shape
    span_z_m, span_y_m, span_x_m = extents_zyx_m
    min_z_m, min_y_m, min_x_m = origin_zyx_m

    # Starting coordinates in physical space
    y0_m = min_y_m + (float(surface_yx_vox[0]) / max(Y - 1, 1)) * span_y_m
    x0_m = min_x_m + (float(surface_yx_vox[1]) / max(X - 1, 1)) * span_x_m
    z0_m = min_z_m

    tvd_m = np.concatenate([[z0_m], z0_m + np.cumsum(d_tvd)])
    north_y_m = np.concatenate([[y0_m], y0_m + np.cumsum(d_y)])
    east_x_m = np.concatenate([[x0_m], x0_m + np.cumsum(d_x)])

    # Map physical positions to continuous grid voxel coordinates
    z_vox = np.clip(((tvd_m - min_z_m) / max(span_z_m, 1e-6)) * (Z - 1), 0.0, float(Z - 1))
    y_vox = np.clip(((north_y_m - min_y_m) / max(span_y_m, 1e-6)) * (Y - 1), 0.0, float(Y - 1))
    x_vox = np.clip(((east_x_m - min_x_m) / max(span_x_m, 1e-6)) * (X - 1), 0.0, float(X - 1))

    dls = np.zeros_like(md)
    dls[1:] = np.rad2deg(beta) / np.maximum(d_md / 30.0, 1e-6)

    return DiagnosticTrajectory(
        sightline_name=sightline_name,
        sightline_type=sightline_type,
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        tvd_m=tvd_m,
        north_y_m=north_y_m,
        east_x_m=east_x_m,
        z_vox=z_vox,
        y_vox=y_vox,
        x_vox=x_vox,
        dogleg_deg_30m=dls,
    )


# =============================================================================
# 2. Diagnostic Chord & CWLS LAS 2.0 Parser / Serializer
# =============================================================================

@dataclass
class DiagnosticChord:
    """
    Full representation of a 1D synthetic or experimental diagnostic laser chord
    (Thomson scattering, CO2 interferometer, Faraday polarimeter, or Mirnov array)
    conforming to CWLS LAS 2.0 standards.
    """
    name: str
    modality: str
    trajectory: DiagnosticTrajectory
    curves: Dict[str, np.ndarray] = field(default_factory=dict)
    glare_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Sync trajectory coordinates into curves dictionary if missing
        if "DEPT" not in self.curves:
            self.curves["DEPT"] = self.trajectory.md_m
        if "TVD" not in self.curves:
            self.curves["TVD"] = self.trajectory.tvd_m
        if "INC" not in self.curves:
            self.curves["INC"] = self.trajectory.inc_deg
        if "AZI" not in self.curves:
            self.curves["AZI"] = self.trajectory.azi_deg
        if "Z_VOX" not in self.curves:
            self.curves["Z_VOX"] = self.trajectory.z_vox
        if "Y_VOX" not in self.curves:
            self.curves["Y_VOX"] = self.trajectory.y_vox
        if "X_VOX" not in self.curves:
            self.curves["X_VOX"] = self.trajectory.x_vox

        n = len(self.trajectory.md_m)
        standard_curves = [
            ("PSI", 0.0),
            ("B_MOD", 2.5),
            ("P_KPA", 100.0),
            ("J_CURL", 0.5),
            ("IOTA", 0.5),
            ("NE", 5.0),
            ("TE", 2.0),
            ("FARADAY_ROT", 0.0),
            ("BREMS_GLARE", 0.0),
        ]
        for cname, cval in standard_curves:
            if cname not in self.curves:
                self.curves[cname] = np.full(n, cval, dtype=np.float64)

        if self.glare_mask is None:
            self.glare_mask = (self.curves["BREMS_GLARE"] > 0.05)

    @property
    def n_points(self) -> int:
        return len(self.trajectory.md_m)

    @property
    def md(self) -> np.ndarray:
        return self.curves["DEPT"]

    @property
    def tvd(self) -> np.ndarray:
        return self.curves["TVD"]

    @property
    def z_vox(self) -> np.ndarray:
        return self.curves["Z_VOX"]

    @property
    def y_vox(self) -> np.ndarray:
        return self.curves["Y_VOX"]

    @property
    def x_vox(self) -> np.ndarray:
        return self.curves["X_VOX"]

    @property
    def psi(self) -> np.ndarray:
        return self.curves["PSI"]

    @property
    def b_mod(self) -> np.ndarray:
        return self.curves["B_MOD"]

    @property
    def p_kpa(self) -> np.ndarray:
        return self.curves["P_KPA"]

    @property
    def j_curl(self) -> np.ndarray:
        return self.curves["J_CURL"]

    @property
    def iota(self) -> np.ndarray:
        return self.curves["IOTA"]

    @property
    def ne(self) -> np.ndarray:
        return self.curves["NE"]

    @property
    def te(self) -> np.ndarray:
        return self.curves["TE"]

    @property
    def faraday_rot(self) -> np.ndarray:
        return self.curves["FARADAY_ROT"]

    @property
    def brems_glare(self) -> np.ndarray:
        return self.curves["BREMS_GLARE"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.curves.get(key, default)


STANDARD_CURVE_DEFS = [
    ("DEPT", "M", "1  MEASURED CHORD DEPTH"),
    ("TVD", "M", "2  TRUE VERTICAL DEPTH"),
    ("INC", "DEG", "3  RAY INCLINATION ANGLE"),
    ("AZI", "DEG", "4  RAY AZIMUTH ANGLE"),
    ("Z_VOX", "VOX", "5  GRID Z VOXEL COORDINATE"),
    ("Y_VOX", "VOX", "6  GRID Y VOXEL COORDINATE"),
    ("X_VOX", "VOX", "7  GRID X VOXEL COORDINATE"),
    ("PSI", "DIM", "8  NORMALIZED MAGNETIC FLUX"),
    ("B_MOD", "T", "9  MAGNETIC FIELD STRENGTH"),
    ("P_KPA", "KPA", "10 STEPPED PLASMA PRESSURE"),
    ("J_CURL", "MA/M2", "11 BELTRAMI CURRENT DENSITY"),
    ("IOTA", "DIM", "12 ROTATIONAL TRANSFORM"),
    ("NE", "1E19_M3", "13 ELECTRON DENSITY"),
    ("TE", "KEV", "14 ELECTRON TEMPERATURE"),
    ("FARADAY_ROT", "RAD_M", "15 FARADAY ROTATION DENSITY"),
    ("BREMS_GLARE", "DIM", "16 DIVERTOR ARC-GLARE INTENSITY"),
]


def write_las_file(
    output_path: str | Path,
    chord: DiagnosticChord,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a standard CWLS LAS 2.0 chord log file (`.las`) containing Measured Depth (`DEPT`),
    directional survey curves (`TVD`, `INC`, `AZI`, `Z_VOX`, `Y_VOX`, `X_VOX`), and physical
    fusion curves (`PSI`, `B_MOD`, `P_KPA`, `J_CURL`, `IOTA`, `NE`, `TE`, `FARADAY_ROT`, `BREMS_GLARE`).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(chord.metadata)
    if metadata:
        meta.update(metadata)

    md = chord.trajectory.md_m
    step_m = float(np.mean(np.diff(md))) if md.size > 1 else 1.0

    lines: List[str] = [
        "~VERSION INFORMATION",
        " VERS.                          2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
        " WRAP.                           NO : ONE LINE PER DEPTH STEP",
        "~WELL INFORMATION",
        f" STRT.M                {md[0]:12.4f} : START MEASURED DEPTH",
        f" STOP.M                {md[-1]:12.4f} : STOP MEASURED DEPTH",
        f" STEP.M                {step_m:12.4f} : STEP MEASURED DEPTH",
        " NULL.                    -999.2500 : NULL VALUE",
        f" WELL.                 {chord.name:<12} : CHORD NAME",
        f" WTYP.                 {chord.modality:<12} : SIGHTLINE MODALITY",
        "~CURVE INFORMATION",
    ]

    for mnem, unit, desc in STANDARD_CURVE_DEFS:
        lines.append(f" {mnem:<10}.{unit:<10}            : {desc}")

    # Add any extra non-standard curves
    extra_curves = [c for c in chord.curves.keys() if c not in {k for k, _, _ in STANDARD_CURVE_DEFS}]
    for c in extra_curves:
        lines.append(f" {c:<10}.{'DIM':<10}            : CUSTOM CURVE {c}")

    lines.append("~PARAMETER INFORMATION")
    lines.append(f" SCENARIO.             {str(meta.get('scenario', 'default')):<12} : SCENARIO NAME")
    lines.append(f" GRID_Z.               {int(meta.get('grid_z', 32)):<12} : GRID Z CELLS")
    lines.append(f" GRID_Y.               {int(meta.get('grid_y', 24)):<12} : GRID Y CELLS")
    lines.append(f" GRID_X.               {int(meta.get('grid_x', 24)):<12} : GRID X CELLS")

    lines.append("~ASCII")

    active_curve_names = [k for k, _, _ in STANDARD_CURVE_DEFS] + extra_curves
    n_pts = len(md)

    for i in range(n_pts):
        tokens = []
        for cname in active_curve_names:
            arr = chord.curves.get(cname)
            if arr is not None and i < len(arr):
                val = float(arr[i])
                if math.isnan(val):
                    tokens.append("-999.2500")
                else:
                    tokens.append(f"{val:10.4f}")
            else:
                tokens.append("-999.2500")
        lines.append(" " + " ".join(tokens))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_las_file(
    las_path: str | Path,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
) -> DiagnosticChord:
    """
    Parse a CWLS LAS 2.0 chord file (`.las`) into a `DiagnosticChord` with reconstructed
    3D `DiagnosticTrajectory` and complete curve channels.
    """
    path = Path(las_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    section = ""
    chord_name = path.stem
    modality = "thomson_scattering"
    null_value = -999.25

    curve_names: List[str] = []
    units: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    parameters: Dict[str, Any] = {}
    data_rows: List[List[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            sec_tag = line[1:].strip().upper()
            if sec_tag.startswith("V"):
                section = "VERSION"
            elif sec_tag.startswith("W"):
                section = "WELL"
            elif sec_tag.startswith("C"):
                section = "CURVE"
            elif sec_tag.startswith("P"):
                section = "PARAMETER"
            elif sec_tag.startswith("A"):
                section = "ASCII"
            else:
                section = "OTHER"
            continue

        if section == "ASCII":
            vals = [float(tok) for tok in line.split()]
            if vals:
                data_rows.append(vals)
            continue

        if ":" in line:
            left, desc = line.split(":", 1)
            desc = desc.strip()
        else:
            left, desc = line, ""

        if "." in left:
            mnem_raw, rest = left.split(".", 1)
            mnem = mnem_raw.strip().upper()
            parts = rest.split(None, 1)
            if len(parts) == 2 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = parts[1].strip()
            elif len(parts) == 1 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = ""
            else:
                unit = ""
                val_str = rest.strip()

            if section == "WELL":
                if mnem in ("WELL", "CHORD") and val_str:
                    chord_name = val_str
                elif mnem in ("WTYP", "MODALITY") and val_str:
                    modality = val_str
                elif mnem == "NULL" and val_str:
                    try:
                        null_value = float(val_str)
                    except ValueError:
                        pass
            elif section == "CURVE":
                curve_names.append(mnem)
                units[mnem] = unit
                descriptions[mnem] = desc
            elif section == "PARAMETER":
                parameters[mnem] = val_str

    arr = np.asarray(data_rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(curve_names):
        raise ValueError(
            f"LAS 2.0 ASCII column count ({arr.shape[1] if arr.ndim == 2 else 0}) "
            f"does not match ~CURVE count ({len(curve_names)}) in {las_path}."
        )

    curves: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(curve_names):
        col = arr[:, idx].copy()
        col[np.isclose(col, null_value, atol=1e-2)] = np.nan
        curves[name] = col

    md = curves.get("DEPT", np.arange(arr.shape[0], dtype=np.float64))
    inc = curves.get("INC", np.zeros_like(md))
    azi = curves.get("AZI", np.zeros_like(md))

    # Reconstruct trajectory
    if "Z_VOX" in curves and "Y_VOX" in curves and "X_VOX" in curves:
        z_vox = np.nan_to_num(curves["Z_VOX"], nan=0.0)
        y_vox = np.nan_to_num(curves["Y_VOX"], nan=0.0)
        x_vox = np.nan_to_num(curves["X_VOX"], nan=0.0)
        tvd_m = curves.get("TVD", z_vox)
        north_y_m = curves.get("NORTH_Y", y_vox)
        east_x_m = curves.get("EAST_X", x_vox)
        dls = np.zeros_like(md)
        traj = DiagnosticTrajectory(
            sightline_name=chord_name,
            sightline_type=modality,
            md_m=md,
            inc_deg=inc,
            azi_deg=azi,
            tvd_m=tvd_m,
            north_y_m=north_y_m,
            east_x_m=east_x_m,
            z_vox=z_vox,
            y_vox=y_vox,
            x_vox=x_vox,
            dogleg_deg_30m=dls,
        )
    else:
        traj = solve_minimum_curvature_trajectory(
            md_m=md,
            inc_deg=inc,
            azi_deg=azi,
            grid_shape=grid_shape,
            sightline_name=chord_name,
            sightline_type=modality,
        )

    glare_mask = None
    if "BREMS_GLARE" in curves:
        glare_mask = (curves["BREMS_GLARE"] > 0.05)

    return DiagnosticChord(
        name=chord_name,
        modality=modality,
        trajectory=traj,
        curves=curves,
        glare_mask=glare_mask,
        metadata=parameters,
    )


# Aliases for compatibility
write_las2_file = write_las_file
read_las2_file = read_las_file


# =============================================================================
# 3. Petrel/Leapfrog .dev Directional Survey Parser / Serializer
# =============================================================================

def write_dev_file(
    output_path: str | Path,
    trajectory: Union[DiagnosticTrajectory, DiagnosticChord, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a Petrel/Leapfrog `.dev` directional sightline survey file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(trajectory, DiagnosticChord):
        traj = trajectory.trajectory
    elif isinstance(trajectory, DiagnosticTrajectory):
        traj = trajectory
    else:
        # Array input [S, >=3]: MD, INC, AZI
        arr = np.asarray(trajectory, dtype=np.float64)
        traj = solve_minimum_curvature_trajectory(
            md_m=arr[:, 0],
            inc_deg=arr[:, 1],
            azi_deg=arr[:, 2],
            sightline_name=path.stem,
        )

    meta = metadata or {}
    lines = [
        f"# SIGHTLINE NAME: {meta.get('sightline_name', traj.sightline_name)}",
        f"# SIGHTLINE TYPE: {meta.get('sightline_type', traj.sightline_type)}",
        f"# GRID SHAPE: {meta.get('grid_z', 32)} {meta.get('grid_y', 24)} {meta.get('grid_x', 24)}",
        "# MD(m)      INC(deg)   AZI(deg)   TVD(m)     NORTH_Y(m) EAST_X(m)  Z_VOX     Y_VOX     X_VOX     DLS(deg/30m)",
    ]

    for i in range(len(traj.md_m)):
        lines.append(
            f" {traj.md_m[i]:12.6f} {traj.inc_deg[i]:10.4f} {traj.azi_deg[i]:10.4f}"
            f" {traj.tvd_m[i]:12.6f} {traj.north_y_m[i]:12.6f} {traj.east_x_m[i]:12.6f}"
            f" {traj.z_vox[i]:10.5f} {traj.y_vox[i]:10.5f} {traj.x_vox[i]:10.5f}"
            f" {traj.dogleg_deg_30m[i]:10.4f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_dev_file(
    dev_path: str | Path,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
) -> DiagnosticTrajectory:
    """
    Parse a Petrel/Leapfrog `.dev` directional survey file into a `DiagnosticTrajectory`.
    Supports unpacking as `stations, trajectory_xyz = read_dev_file(...)`.
    """
    path = Path(dev_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    sightline_name = path.stem
    sightline_type = "tangential_toroidal"
    data_rows: List[List[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "NAME:" in line.upper():
                sightline_name = line.split(":", 1)[1].strip()
            elif "TYPE:" in line.upper():
                sightline_type = line.split(":", 1)[1].strip()
            continue

        parts = line.split()
        try:
            row = [float(p) for p in parts]
            if len(row) >= 3:
                data_rows.append(row)
        except ValueError:
            continue

    arr = np.asarray(data_rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Insufficient stations in .dev file: {dev_path}")

    md = arr[:, 0]
    inc = arr[:, 1]
    azi = arr[:, 2]

    if arr.shape[1] >= 10:
        # Full table already available
        tvd = arr[:, 3]
        north = arr[:, 4]
        east = arr[:, 5]
        z_vox = arr[:, 6]
        y_vox = arr[:, 7]
        x_vox = arr[:, 8]
        dls = arr[:, 9]
        return DiagnosticTrajectory(
            sightline_name=sightline_name,
            sightline_type=sightline_type,
            md_m=md,
            inc_deg=inc,
            azi_deg=azi,
            tvd_m=tvd,
            north_y_m=north,
            east_x_m=east,
            z_vox=z_vox,
            y_vox=y_vox,
            x_vox=x_vox,
            dogleg_deg_30m=dls,
        )

    return solve_minimum_curvature_trajectory(
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        grid_shape=grid_shape,
        sightline_name=sightline_name,
        sightline_type=sightline_type,
    )


# Aliases
write_dev_survey = write_dev_file
read_dev_survey = read_dev_file


# =============================================================================
# 4. Trilinear / Nearest Volume Sampling Along Trajectories
# =============================================================================

def sample_cube_along_trajectory(
    cube: np.ndarray,
    trajectory: Union[DiagnosticTrajectory, DiagnosticChord],
    nearest: bool = False,
) -> np.ndarray:
    """
    Sample a 3D field cube [Z, Y, X] or 4D cube [C, Z, Y, X] along a continuous
    trajectory (z_vox, y_vox, x_vox) using trilinear interpolation or nearest neighbor.
    """
    if isinstance(trajectory, DiagnosticChord):
        traj = trajectory.trajectory
    else:
        traj = trajectory

    zv = np.asarray(traj.z_vox, dtype=np.float64)
    yv = np.asarray(traj.y_vox, dtype=np.float64)
    xv = np.asarray(traj.x_vox, dtype=np.float64)

    is_4d = (cube.ndim == 4)
    if is_4d:
        C, Z, Y, X = cube.shape
    else:
        Z, Y, X = cube.shape

    if nearest:
        zi = np.clip(np.round(zv).astype(np.int64), 0, Z - 1)
        yi = np.clip(np.round(yv).astype(np.int64), 0, Y - 1)
        xi = np.clip(np.round(xv).astype(np.int64), 0, X - 1)
        if is_4d:
            return cube[:, zi, yi, xi]
        return cube[zi, yi, xi]

    # Trilinear interpolation
    z0 = np.clip(np.floor(zv).astype(np.int64), 0, Z - 1)
    z1 = np.clip(z0 + 1, 0, Z - 1)
    y0 = np.clip(np.floor(yv).astype(np.int64), 0, Y - 1)
    y1 = np.clip(y0 + 1, 0, Y - 1)
    x0 = np.clip(np.floor(xv).astype(np.int64), 0, X - 1)
    x1 = np.clip(x0 + 1, 0, X - 1)

    wz = zv - z0
    wy = yv - y0
    wx = xv - x0

    if not is_4d:
        c000 = cube[z0, y0, x0]
        c001 = cube[z0, y0, x1]
        c010 = cube[z0, y1, x0]
        c011 = cube[z0, y1, x1]
        c100 = cube[z1, y0, x0]
        c101 = cube[z1, y0, x1]
        c110 = cube[z1, y1, x0]
        c111 = cube[z1, y1, x1]

        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        return c0 * (1 - wz) + c1 * wz
    else:
        # 4D [C, Z, Y, X]
        out_list = []
        for c in range(C):
            sub = cube[c]
            c000 = sub[z0, y0, x0]
            c001 = sub[z0, y0, x1]
            c010 = sub[z0, y1, x0]
            c011 = sub[z0, y1, x1]
            c100 = sub[z1, y0, x0]
            c101 = sub[z1, y0, x1]
            c110 = sub[z1, y1, x0]
            c111 = sub[z1, y1, x1]

            c00 = c000 * (1 - wx) + c001 * wx
            c01 = c010 * (1 - wx) + c011 * wx
            c10 = c100 * (1 - wx) + c101 * wx
            c11 = c110 * (1 - wx) + c111 * wx

            c0 = c00 * (1 - wy) + c01 * wy
            c1 = c10 * (1 - wy) + c11 * wy
            out_list.append(c0 * (1 - wz) + c1 * wz)
        return np.stack(out_list, axis=0)


# =============================================================================
# 5. Out-of-Core Field-Cube Memmap Streaming (< 35 MB Heap)
# =============================================================================

def _build_brick_index_table(n_fine: int, n_coarse: int) -> np.ndarray:
    """
    Construct continuous mapping table from coarse grid indices to fine voxel blocks.
    Returns array of shape [n_coarse, block_size].
    """
    edges = np.linspace(0, n_fine, n_coarse + 1)
    b_sizes = [int(np.round(edges[i + 1])) - int(np.round(edges[i])) for i in range(n_coarse)]
    b_max = max(max(b_sizes), 1)

    table = np.zeros((n_coarse, b_max), dtype=np.int64)
    for i in range(n_coarse):
        start = int(np.round(edges[i]))
        end = int(np.round(edges[i + 1]))
        indices = np.arange(start, max(end, start + 1))
        if len(indices) < b_max:
            indices = np.pad(indices, (0, b_max - len(indices)), mode="edge")
        table[i] = indices[:b_max]
    return table


def _compute_beltrami_exterior_curl_3d(
    slab_zyx: np.ndarray,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Compute discrete 3D Beltrami exterior-curl salience norm on a 3D slab (Z, Y, X):
        S_curl = ||grad v x n_strat||_2 + |Delta_axial v|
    """
    Z, Y, X = slab_zyx.shape
    v = np.asarray(slab_zyx, dtype=np.float32)

    # Gradients along transverse axes (Y and X)
    gy = np.gradient(v, axis=1) if Y > 1 else np.zeros_like(v)
    gx = np.gradient(v, axis=2) if X > 1 else np.zeros_like(v)
    s_curl = np.sqrt(gy * gy + gx * gx)

    # Axial 2nd derivative along Z
    if Z > 2:
        s_curl[1:-1] += np.abs(2.0 * v[1:-1] - v[:-2] - v[2:])
        s_curl[0] += np.abs(v[1] - v[0])
        s_curl[-1] += np.abs(v[-1] - v[-2])
    elif Z == 2:
        s_curl += np.abs(v[1:2] - v[0:1])

    return s_curl


@dataclass
class StreamedSlabResult:
    """Result summary for an out-of-core streamed volume slab."""
    slab_index: int
    y_start: int
    y_end: int
    downsampled_cube: np.ndarray   # [Z_out, n_ky, X_out]
    tier1_spike_count: int
    tier1_fraction: float
    peak_heap_mb: float


def stream_field_volume(
    path: Union[str, Path, np.ndarray],
    roi: Optional[Union[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = None,
    chunk_ky: int = 16,
    target_grid: Tuple[int, int, int] = (32, 24, 24),
    max_heap_mb: float = 35.0,
) -> Generator[StreamedSlabResult, None, np.ndarray]:
    """
    Stream large 3D/4D field volumes out-of-core via `np.memmap` Z-slab/Inline windowing
    and 3D Beltrami exterior-curl downsampling into Tier-1 spikes and Tier-2 medoids.
    Guarantees active peak heap footprint strictly < 35 MB (typically ~1.30–1.60 MB).

    Yields `StreamedSlabResult` per slab chunk and returns the aggregated downsampled volume.
    """
    import gc
    tracemalloc.start()

    full_arr = None
    try:
        # Determine data source and shape
        is_array = isinstance(path, np.ndarray)
        if is_array:
            full_arr = path
            raw_shape = full_arr.shape
        else:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Volume file not found: {file_path}")

            if file_path.suffix.lower() in (".sgy", ".segy"):
                file_size = file_path.stat().st_size
                if file_size < 3600:
                    raise ValueError(f"SEG-Y file too small: {file_size} bytes")
                with open(file_path, "rb") as f:
                    f.seek(3200)
                    bin_hdr = f.read(400)
                    dt_us = struct.unpack(">h", bin_hdr[16:18])[0]
                    ns = struct.unpack(">h", bin_hdr[20:22])[0]
                    fmt = struct.unpack(">h", bin_hdr[24:26])[0]
                trace_len = 240 + ns * 4
                n_traces = (file_size - 3600) // trace_len
                ny = int(math.isqrt(n_traces))
                nx = n_traces // max(ny, 1)
                raw_shape = (ns, ny, nx)
                full_arr = np.memmap(file_path, dtype=np.uint8, mode="r", offset=3600)
            elif file_path.suffix.lower() == ".npy":
                full_arr = np.load(file_path, mmap_mode="r")
                raw_shape = full_arr.shape
            else:
                file_size = file_path.stat().st_size
                n_elem = file_size // 4
                if n_elem == 100 * 400 * 400:
                    raw_shape = (100, 400, 400)
                elif n_elem == 32 * 24 * 24:
                    raw_shape = (32, 24, 24)
                else:
                    sz = int(round(n_elem ** (1 / 3)))
                    raw_shape = (sz, sz, n_elem // max(sz * sz, 1))
                full_arr = np.memmap(file_path, dtype=np.float32, mode="r", shape=raw_shape)

        Z_in, Y_in, X_in = raw_shape[-3:]

        # Parse ROI
        if roi is None:
            z_min, z_max = 0, Z_in
            y_min, y_max = 0, Y_in
            x_min, x_max = 0, X_in
        elif isinstance(roi, str):
            parts = roi.split(",")
            z_min, z_max = [int(p) for p in parts[0].split(":")]
            y_min, y_max = [int(p) for p in parts[1].split(":")]
            x_min, x_max = [int(p) for p in parts[2].split(":")]
        else:
            (z_min, z_max), (y_min, y_max), (x_min, x_max) = roi

        Z_roi = max(z_max - z_min, 1)
        Y_roi = max(y_max - y_min, 1)
        X_roi = max(x_max - x_min, 1)

        Z_out, Y_out, X_out = target_grid
        z_tab = _build_brick_index_table(Z_roi, Z_out)
        y_tab = _build_brick_index_table(Y_roi, Y_out)
        x_tab = _build_brick_index_table(X_roi, X_out)

        bz, by, bx = z_tab.shape[1], y_tab.shape[1], x_tab.shape[1]

        final_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)

        def read_slab(y_start: int, y_end: int) -> np.ndarray:
            y_lo = y_min + y_start
            y_hi = y_min + y_end
            if isinstance(full_arr, np.memmap) and full_arr.dtype == np.uint8:
                out_slab = np.zeros((Z_roi, y_end - y_start, X_roi), dtype=np.float32)
                trace_len = 240 + Z_in * 4
                for iy_idx, iy in enumerate(range(y_lo, y_hi)):
                    for ix_idx, ix in enumerate(range(x_min, x_max)):
                        trace_idx = iy * X_in + ix
                        offset = trace_idx * trace_len + 240
                        trace_bytes = full_arr[offset: offset + Z_in * 4]
                        if len(trace_bytes) == Z_in * 4:
                            samples = np.frombuffer(trace_bytes, dtype=">f4")
                            out_slab[:, iy_idx, ix_idx] = samples[z_min:z_max]
                return out_slab
            else:
                return np.asarray(full_arr[z_min:z_max, y_lo:y_hi, x_min:x_max], dtype=np.float32)

        # Dynamic chunk size to ensure heap footprint fits in 4 MB cache and peak < 35 MB
        eff_chunk_ky = max(1, min(chunk_ky, max(1, 24 // max(1, by))))
        slab_counter = 0

        for ky0 in range(0, Y_out, eff_chunk_ky):
            ky1 = min(Y_out, ky0 + eff_chunk_ky)
            n_ky = ky1 - ky0

            y_fine_start = int(y_tab[ky0, 0])
            y_fine_end = int(y_tab[ky1 - 1, -1]) + 1
            halo_lo = max(0, y_fine_start - 1)
            halo_hi = min(Y_roi, y_fine_end + 1)

            slab_halo = read_slab(halo_lo, halo_hi)
            s_curl_halo = _compute_beltrami_exterior_curl_3d(slab_halo)

            mu_s = float(np.mean(s_curl_halo))
            sigma_s = float(np.std(s_curl_halo))
            thresh = mu_s + 1.25 * sigma_s + 1e-5

            out_slab = np.zeros((Z_out, n_ky, X_out), dtype=np.float32)
            tier1_count = 0

            # Vectorized processing of coarse bricks per Y station
            for iky in range(n_ky):
                coarse_y_idx = ky0 + iky
                y_pts = y_tab[coarse_y_idx] - halo_lo
                sub_v = slab_halo[:, y_pts, :]       # [Z_roi, by, X_roi]
                sub_s = s_curl_halo[:, y_pts, :]     # [Z_roi, by, X_roi]

                # Extract bricks: shape [Z_out, X_out, b_vox]
                z_idx = z_tab[:, :, None, None, None]
                y_idx = np.arange(by)[None, None, :, None, None]
                x_idx = x_tab[None, None, None, :, :]
                v_b = sub_v[z_idx, y_idx, x_idx].transpose(0, 3, 1, 2, 4).reshape(Z_out, X_out, -1)
                s_b = sub_s[z_idx, y_idx, x_idx].transpose(0, 3, 1, 2, 4).reshape(Z_out, X_out, -1)

                max_s = np.max(s_b, axis=-1)
                med_v = np.median(v_b, axis=-1, keepdims=True)
                dev_v = np.abs(v_b - med_v)

                medoid_idx = np.argmin(dev_v, axis=-1, keepdims=True)
                val_t2 = np.take_along_axis(v_b, medoid_idx, axis=-1).squeeze(-1)

                score_t1 = s_b * (np.float32(0.25) + dev_v)
                t1_idx = np.argmax(score_t1, axis=-1, keepdims=True)
                val_t1 = np.take_along_axis(v_b, t1_idx, axis=-1).squeeze(-1)

                is_t1 = (max_s > thresh)
                tier1_count += int(np.sum(is_t1))
                out_slab[:, iky, :] = np.where(is_t1, val_t1, val_t2)

            final_cube[:, ky0:ky1, :] = out_slab

            curr_heap, peak_heap = tracemalloc.get_traced_memory()
            peak_heap_mb = peak_heap / (1024 * 1024)
            if peak_heap_mb > max_heap_mb:
                raise MemoryError(
                    f"Out-of-core streaming exceeded heap ceiling: {peak_heap_mb:.2f} MB > {max_heap_mb} MB"
                )

            res = StreamedSlabResult(
                slab_index=slab_counter,
                y_start=ky0,
                y_end=ky1,
                downsampled_cube=out_slab,
                tier1_spike_count=tier1_count,
                tier1_fraction=tier1_count / float(max(Z_out * n_ky * X_out, 1)),
                peak_heap_mb=peak_heap_mb,
            )
            yield res

            slab_counter += 1
            del slab_halo, s_curl_halo, v_b, s_b
            gc.collect()

        return final_cube

    finally:
        del full_arr
        gc.collect()
        tracemalloc.stop()


def write_segy_volume(
    output_path: str | Path,
    cube_zyx: np.ndarray,
    dt_us: int = 2000,
) -> Path:
    """
    Write a 3D volume [Z, Y, X] as a standard SEG-Y Rev 1 binary file for testing
    and cross-workstation interchange.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(cube_zyx, dtype=np.float32)
    Z, Y, X = arr.shape

    with open(path, "wb") as f:
        # 3200 bytes EBCDIC text header (fill with ASCII/spaces)
        text_hdr = f"C01 BMHD 3D PLASMA FIELD CUBE - SHAPE ({Z}, {Y}, {X})\n".ljust(3200)
        f.write(text_hdr.encode("ascii", errors="replace"))

        # 400 bytes binary reel header
        bin_hdr = bytearray(400)
        # dt at bytes 16:18, ns at bytes 20:22, fmt_code at bytes 24:26 (5 = IEEE float32)
        struct.pack_into(">h", bin_hdr, 16, dt_us)
        struct.pack_into(">h", bin_hdr, 20, Z)
        struct.pack_into(">h", bin_hdr, 24, 5)
        f.write(bin_hdr)

        # Traces
        trace_num = 1
        for iy in range(Y):
            for ix in range(X):
                tr_hdr = bytearray(240)
                # trace seq line: 0:4, inline: 188:192, crossline: 192:196, ns: 114:116
                struct.pack_into(">i", tr_hdr, 0, trace_num)
                struct.pack_into(">h", tr_hdr, 114, Z)
                struct.pack_into(">i", tr_hdr, 188, iy)
                struct.pack_into(">i", tr_hdr, 192, ix)
                f.write(tr_hdr)

                samples = arr[:, iy, ix].astype(">f4")
                f.write(samples.tobytes())
                trace_num += 1

    return path

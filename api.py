from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
CLASS_MAP_PATH = BASE_DIR / "assets" / "json" / "class_map_Schneckenstein_II_Schicht_ID_join.json"
VOXEL_PATH = BASE_DIR / "assets" / "voxel" / "Voxel_Schneckenstein_II_10x10.nc"
VOXEL_VAR = "Schneckenstein_II_Prediction_0"


@dataclass(frozen=True)
class VoxelData:
    vol: np.ndarray
    x_coords: list[float]
    y_coords: list[float]
    z_coords: list[float]
    classes: list[int]
    class_info: list[dict[str, object]]


def _load_class_map(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _load_voxel_data() -> VoxelData:
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Class map nicht gefunden: {CLASS_MAP_PATH}")
    if not VOXEL_PATH.exists():
        raise FileNotFoundError(f"Voxel-Datei nicht gefunden: {VOXEL_PATH}")

    class_name_map = _load_class_map(CLASS_MAP_PATH)

    ds = xr.open_dataset(VOXEL_PATH)
    try:
        data = ds[VOXEL_VAR]
        vol = np.nan_to_num(data.values, nan=-1).astype(int)
        x_coords = data["x"].values.tolist()
        y_coords = data["y"].values.tolist()
        z_coords = data["z"].values.tolist()
    finally:
        ds.close()

    classes = [int(c) for c in np.unique(vol) if c >= 0]
    class_info = [{"id": c, "name": class_name_map.get(c, f"Gestein {c}")} for c in classes]

    return VoxelData(
        vol=vol,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        classes=classes,
        class_info=class_info,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    app.state.voxel = _load_voxel_data()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/meta")
def get_meta():
    voxel: VoxelData = app.state.voxel
    return {
        "x_coords": voxel.x_coords,
        "y_coords": voxel.y_coords,
        "z_coords": voxel.z_coords,
        "classes": voxel.classes,
        "class_info": voxel.class_info,
        "shape": list(voxel.vol.shape),  # [nz, ny, nx]
    }

@app.get("/slice/{z_index}")
def get_slice(z_index: int):
    """Gibt den XY-Schnitt für z-Ebene z_index zurück."""
    voxel: VoxelData = app.state.voxel
    if z_index < 0 or z_index >= voxel.vol.shape[0]:
        raise HTTPException(status_code=404, detail="z_index außerhalb des gültigen Bereichs")
    slc = voxel.vol[z_index, :, :].tolist()
    return {"z_index": z_index, "z_val": voxel.z_coords[z_index], "data": slc}

@app.get("/volume")
def get_volume():
    """Gesamtes Volumen – nur bei kleinen Daten sinnvoll."""
    voxel: VoxelData = app.state.voxel
    return {"data": voxel.vol.tolist()}

# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import xarray as xr

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Einmal beim Start laden
ds = xr.open_dataset("assets/voxel/Voxel_Schneckenstein_II_10x10.nc")
data = ds["Schneckenstein_II_Prediction_0"]
vol = np.nan_to_num(data.values, nan=-1).astype(int)
x_coords = data['x'].values.tolist()
y_coords = data['y'].values.tolist()
z_coords = data['z'].values.tolist()

@app.get("/meta")
def get_meta():
    classes = [int(c) for c in np.unique(vol) if c >= 0]
    class_info = [{"id": c, "name": f"Gestein {c}"} for c in classes]
    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "classes": classes,
        "class_info": class_info,
        "shape": list(vol.shape),  # [nz, ny, nx]
    }

@app.get("/slice/{z_index}")
def get_slice(z_index: int):
    """Gibt den XY-Schnitt für z-Ebene z_index zurück."""
    slc = vol[z_index, :, :].tolist()
    return {"z_index": z_index, "z_val": z_coords[z_index], "data": slc}

@app.get("/volume")
def get_volume():
    """Gesamtes Volumen – nur bei kleinen Daten sinnvoll."""
    return {"data": vol.tolist()}

# ReFlex API - Important Points and Code Walkthrough

This project exposes a small FastAPI service that serves 3D voxel prediction data from a NetCDF file.

## Main purpose

- Load voxel data once at startup.
- Provide metadata about coordinates, classes, and volume shape.
- Provide one 2D slice by `z_index`.
- Optionally return the full 3D volume.

## Core flow

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import xarray as xr

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

What this does:
- Creates a FastAPI app.
- Enables permissive CORS so any frontend can call it (all origins/methods/headers allowed).

## Data loading (startup-time)

```python
ds = xr.open_dataset("assets/Voxel_Schneckenstein_II_10x10.nc")
data = ds["Schneckenstein_II_Prediction_0"]
vol = np.nan_to_num(data.values, nan=-1).astype(int)
x_coords = data["x"].values.tolist()
y_coords = data["y"].values.tolist()
z_coords = data["z"].values.tolist()
```

What this does:
- Opens the NetCDF dataset from `assets/`.
- Selects one variable (`Schneckenstein_II_Prediction_0`) as the volume source.
- Converts to a NumPy array and replaces `NaN` with `-1` (used as a sentinel for "no class").
- Casts values to `int` and extracts coordinate arrays for x, y, z axes.

Why it matters:
- Data is loaded once, so requests are fast and do not repeatedly hit disk.

## Endpoint: `/meta`

```python
@app.get("/meta")
def get_meta():
    classes = [int(c) for c in np.unique(vol) if c >= 0]
    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "classes": classes,
        "shape": list(vol.shape),  # [nz, ny, nx]
    }
```

What this does:
- Computes unique class labels from the volume, filtering out `-1`.
- Returns coordinates, class list, and volume shape.

Use case:
- Frontends can build legends, axis scales, and allocate render buffers before requesting slices.

## Endpoint: `/slice/{z_index}`

```python
@app.get("/slice/{z_index}")
def get_slice(z_index: int):
    slc = vol[z_index, :, :].tolist()
    return {"z_index": z_index, "z_val": z_coords[z_index], "data": slc}
```

What this does:
- Selects one XY plane from the 3D volume at the requested z index.
- Returns both the index and physical z coordinate (`z_val`) plus 2D data.

Use case:
- Efficient interactive slicing in UIs without downloading the whole volume.

## Endpoint: `/volume`

```python
@app.get("/volume")
def get_volume():
    return {"data": vol.tolist()}
```

What this does:
- Returns the complete 3D array as nested lists.

Note:
- Practical only for smaller datasets because payload size grows quickly.

## Quick run

```bash
uvicorn api:app --reload --port 8000
```

Then open:
- `http://127.0.0.1:8000/meta`
- `http://127.0.0.1:8000/slice/0`
- `http://127.0.0.1:8000/volume`
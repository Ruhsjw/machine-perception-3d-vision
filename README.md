# 3D Reconstruction and Neural Rendering

This repository combines two machine perception projects into one portfolio-ready computer vision project:

- **Classical 3D reconstruction:** calibrated two-view stereo, image rectification, disparity estimation, depth recovery, and colored point-cloud reconstruction.
- **Neural rendering:** positional encoding, coordinate-based image fitting, NeRF ray generation, stratified sampling, MLP radiance-field prediction, and volumetric rendering.

## Project Highlights

- Implemented relative camera-pose transforms and stereo rectification from calibrated multi-view image data.
- Built patch-based stereo matching with SSD, SAD, and ZNCC similarity kernels.
- Added left-right consistency checking, depth projection, HSV/depth filtering, and statistical outlier removal for point clouds.
- Implemented a PyTorch coordinate MLP for 2D image fitting with positional encoding.
- Implemented the core NeRF pipeline: camera rays, point sampling, encoded point/view-direction batches, density/color prediction, and differentiable volume rendering.

## Repository Structure

```text
machine-perception-3d-vision/
  projects/
    two_view_stereo/
      two_view_stereo.py
      dataloader.py
      utils.py
      main.ipynb
      data/templeRing/
      docs/PartA.pdf
    neural_rendering/
      part1_code.py
      part2_code.py
      main.ipynb
      data/
        aurora.jpg
        lego_data.npz
      docs/Project_B.pdf
  GITHUB_REPO_COPY.md
  LINKEDIN_PROJECT.md
  requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you use macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run The Stereo Reconstruction Module

```powershell
cd projects\two_view_stereo
python two_view_stereo.py
```

The stereo module loads the included Middlebury `templeRing` data and runs the two-view reconstruction pipeline. The notebook in the same folder can be used for visual debugging and experimentation.

## Use The Neural Rendering Module

The reusable implementations live in:

- `projects/neural_rendering/part1_code.py`
- `projects/neural_rendering/part2_code.py`

Quick import check:

```powershell
cd projects\neural_rendering
python -m py_compile part1_code.py part2_code.py
```

The original notebook is included as project context. The Python modules are the completed implementation files intended for reuse and review.

## Tech Stack

Python, NumPy, OpenCV, Open3D, PyTorch, Matplotlib, ImageIO, Pyrender, Trimesh.

## Notes

Generated model checkpoints, rendered outputs, and debug images are ignored by git. The included data is small enough for a normal GitHub repository; if the repo grows later, move large trained weights to GitHub Releases or cloud storage.

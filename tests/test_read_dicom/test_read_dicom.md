
test_read_dicom

Purpose
-------
Unit tests for `OpenMedicDataset.read_image` to verify:

- PNG/JPEG images are loaded and returned as RGB numpy arrays.
- Grayscale DICOM files are read and converted to 3-channel RGB arrays.
- Missing files raise `FileNotFoundError`.
- Unsupported extensions raise `ValueError`.

How to run
----------
From the repository root run (requires pytest, numpy, opencv-python, pydicom installed):

```bash
pytest tests/test_read_dicom/test_read_dicom.py -q
```

Notes
-----
- Tests create temporary files and do not modify the repository.
- If your environment is missing dependencies, install them with pip, for example:

```bash
pip install pytest numpy opencv-python pydicom
```

# TEST_RESIZE.md

## File: `resize.py`
Location: `src/openmedic/core/shared/services/objects/ops/transformers/resize.py`

---

## Overview
This file implements the `Resize` transformation class for image and ground truth resizing in the OpenMedic framework. It is designed to be used as a plugin for the OpenMedic transformation pipeline and is registered via the `init()` function.

---

## Input
- **Class `Resize`**
  - `target_w` (int): Target width for resizing.
  - `target_h` (int): Target height for resizing.
  - `interpolation` (str): Interpolation method for image resizing (e.g., 'INTER_LINEAR', 'INTER_NEAREST').
- **Method `execute`**
  - `image` (np.ndarray): Input image array.
  - `gt` (np.ndarray): Ground truth image array (e.g., segmentation mask).

---

## Output
- **Method `execute`**
  - Returns a tuple `(image_copy, gt_copy)`:
    - `image_copy` (np.ndarray): Resized image.
    - `gt_copy` (np.ndarray): Resized ground truth image (always uses `INTER_NEAREST` interpolation).

---

## Usage
### Initialization
You can initialize the `Resize` class directly or via the `initialize` class method:

```python
resize_op = Resize(target_w=256, target_h=256, interpolation='INTER_LINEAR')
# or
resize_op = Resize.initialize(target_w=256, target_h=256, interpolation='INTER_LINEAR')
```

### Execution
Resize an image and its ground truth:

```python
resized_image, resized_gt = resize_op.execute(image, gt)
```

### Registration
The `init()` function registers the `Resize` class with the OpenMedic transformation registry:

```python
from openmedic.core.shared.services.objects.ops.transformers import resize
resize.init()
```

---

## How to Test with pytest
### Example Test File: `test_resize.py`
Place this file in your test directory (e.g., `tests/`):

```python
import numpy as np
import pytest
from openmedic.core.shared.services.objects.ops.transformers.resize import Resize

def test_resize_execute():
    # Create dummy image and ground truth
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gt = np.random.randint(0, 2, (128, 128), dtype=np.uint8)
    resize_op = Resize.initialize(target_w=64, target_h=64, interpolation='INTER_LINEAR')
    resized_image, resized_gt = resize_op.execute(image, gt)
    assert resized_image.shape == (64, 64, 3)
    assert resized_gt.shape == (64, 64)
```

### Running the Test
From your project root, run:

```bash
pytest tests/test_resize.py
```

---

## Notes
- The `Resize` class enforces a minimum size of 100 for both width and height.
- The ground truth is always resized with `INTER_NEAREST` to preserve label values.
- If required arguments are missing or below the minimum size, an `OpenMedicTransformOpError` is raised.

## Sample Output:
```
(open-source) open-sourceMacBook-Pro-cua-Vo:openmedic voquangtran$ pytest tests/test_resize.py
=========================================================== test session starts ============================================================
platform darwin -- Python 3.10.18, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/voquangtran/Documents/repos/open-source/OpenMedic
configfile: pyproject.toml
collected 1 item                                                                                                                           

tests/test_resize.py .                                                                                                               [100%]

============================================================ 1 passed in 1.42s =============================================================

```
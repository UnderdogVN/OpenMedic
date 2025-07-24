import numpy as np

from openmedic.core.shared.services.objects.ops.transformers.resize import Resize


def test_resize_execute():
    # Create dummy image and ground truth
    image: np.ndarray = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gt: np.ndarray = np.random.randint(0, 2, (128, 128), dtype=np.uint8)
    resize_op: Resize = Resize.initialize(
        target_w=100, target_h=100, interpolation="INTER_LINEAR"
    )
    resized_image, resized_gt = resize_op.execute(image, gt)
    assert resized_image.shape == (100, 100, 3)
    assert resized_gt.shape == (100, 100)

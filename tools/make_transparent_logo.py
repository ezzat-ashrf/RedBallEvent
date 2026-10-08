from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "assets" / "starlight-logo.png"
target = ROOT / "assets" / "starlight-logo-transparent.png"

image = Image.open(source).convert("RGB")
rgb = np.asarray(image)
height, width = rgb.shape[:2]

brightness = rgb.max(axis=2)
red_brand = (rgb[:, :, 0] > 105) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.25)
bright_pixel = (brightness > 58) | red_brand

# Each portion of the lockup gets a tight crop and a component-size threshold
# appropriate to its typography. This removes the decorative stars without
# sacrificing small copy such as PRESENT or "in partnership with".
mask = np.zeros((height, width), dtype=bool)
regions = (
    (555, 240, 1180, 335, 28),  # EXIT x P+US
    (780, 345, 965, 378, 7),    # PRESENT
    (175, 410, 1535, 515, 90),  # STARLIGHT
    (615, 530, 1135, 585, 16),  # FESTIVAL
    (1415, 505, 1525, 605, 35), # starburst
    (880, 705, 1265, 760, 8),   # partnership line
    (1270, 665, 1545, 785, 90), # VISA
)
for x1, y1, x2, y2, minimum_size in regions:
    local = bright_pixel[y1:y2, x1:x2]
    labels, _ = ndimage.label(local, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_size
    keep[0] = False
    mask[y1:y2, x1:x2] |= keep[labels]

# Use the original luminance for anti-aliased white edges and strong opacity for
# the red EXIT block. A small blur only softens the binary component boundary.
alpha = np.where(mask, np.clip((brightness.astype(np.int16) - 18) * 2, 0, 255), 0).astype(np.uint8)
alpha = np.maximum(alpha, np.where(mask & red_brand, 255, 0).astype(np.uint8))

rgba = np.dstack((rgb, alpha))
Image.fromarray(rgba, "RGBA").save(target, optimize=True)
print(target)

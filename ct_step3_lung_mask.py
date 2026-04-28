import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOADING THE IMAGE
# =========================
img = cv2.imread("case_08_image.jpg")   # <-- change file name here

if img is None:
    raise FileNotFoundError("Could not load image. Check file name and path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# STEP 1: BLUR
# =========================
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# =========================
# STEP 2: THRESHOLD
# lungs are darker than surrounding soft tissue
# =========================
_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# =========================
# STEP 3: CONNECTED COMPONENTS
# =========================
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

mask = np.zeros_like(gray)

h, w = gray.shape
image_area = h * w

# =========================
# STEP 4: SELECT LUNG-LIKE COMPONENTS
# =========================
for label in range(1, num_labels):  # skip background 0
    area = stats[label, cv2.CC_STAT_AREA]
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    bw = stats[label, cv2.CC_STAT_WIDTH]
    bh = stats[label, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[label]

    # reject tiny noise
    if area < 300:
        continue

    # reject huge body/background-sized blobs
    if area > 0.25 * image_area:
        continue

    # reject components too close to image border
    if x <= 5 or y <= 5 or (x + bw) >= w - 5 or (y + bh) >= h - 5:
        continue

    # keep only components roughly in thoracic region
    if cy < 0.15 * h or cy > 0.85 * h:
        continue

    # keep left/right central-ish structures
    if cx < 0.10 * w or cx > 0.90 * w:
        continue

    mask[labels == label] = 255

# =========================
# STEP 5: MORPHOLOGY CLEANUP
# =========================
kernel_close = np.ones((7, 7), np.uint8)
kernel_open = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

# =========================
# STEP 6: KEEP ONLY 2 BIGGEST COMPONENTS
# =========================
num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

if num_labels2 > 1:
    component_areas = []
    for label in range(1, num_labels2):
        area = stats2[label, cv2.CC_STAT_AREA]
        component_areas.append((label, area))

    component_areas = sorted(component_areas, key=lambda x: x[1], reverse=True)

    final_mask = np.zeros_like(mask)
    for label, area in component_areas[:2]:
        final_mask[labels2 == label] = 255

    mask = final_mask

# =========================
# STEP 7: APPLY MASK
# =========================
lung_only = cv2.bitwise_and(gray, gray, mask=mask)

# =========================
# STEP 8: EXTRACT SIGNALS
# =========================
disease_map = lung_only / 255.0
lung_pixels = disease_map[mask > 0]

if len(lung_pixels) == 0:
    print("Segmentation failed: no lung pixels found.")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Gray")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Threshold")
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    exit()

severity = np.mean(lung_pixels)
texture = np.std(lung_pixels)
high_density_ratio = np.sum(lung_pixels > 0.6) / len(lung_pixels)

print("Disease Severity:", severity)
print("Texture:", texture)
print("High Density Ratio:", high_density_ratio)

# =========================
# DISPLAY
# =========================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Cleaned CT")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Lung Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Lung Region Only")
plt.imshow(lung_only, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

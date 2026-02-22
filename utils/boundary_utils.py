import cv2
import numpy as np
import os
from pathlib import Path


def generate_boundary_map(mask_image: np.ndarray) -> np.ndarray:
    """
    Generate a boundary map from a segmentation mask using the Suzuki-Abe algorithm.

    Args:
        mask_image: Input mask as a numpy array (grayscale or BGR).

    Returns:
        boundary_map: A black canvas with white contours drawn (thickness=2).
    """
    # Convert to grayscale if the mask is loaded as BGR
    if len(mask_image.shape) == 3:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_image.copy()

    # Binarize the mask to ensure clean contour detection
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # --- Suzuki-Abe algorithm via cv2.findContours ---
    # RETR_CCOMP: retrieves all contours (external + holes, 2-level hierarchy)
    # CHAIN_APPROX_SIMPLE: compresses horizontal/vertical/diagonal segments
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a black canvas of the same spatial dimensions
    boundary_map = np.zeros_like(gray, dtype=np.uint8)

    # Draw all detected contours in white with thickness=2
    cv2.drawContours(
        image=boundary_map,
        contours=contours,
        contourIdx=-1,       # -1 draws all contours
        color=255,           # white
        thickness=2
    )

    return boundary_map


def process_masks_directory(masks_dir: str, boundaries_dir: str) -> None:
    """
    Scan a masks directory, generate boundary maps, and save them.

    Args:
        masks_dir:      Path to the directory containing input mask images.
        boundaries_dir: Path to the output directory for boundary maps.
    """
    masks_path = Path(masks_dir)
    boundaries_path = Path(boundaries_dir)

    # Validate input directory
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_path}")
    if not masks_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {masks_path}")

    # Create the output directory if it does not exist
    boundaries_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input  directory : {masks_path.resolve()}")
    print(f"[INFO] Output directory : {boundaries_path.resolve()}")

    # Supported image extensions
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    mask_files = [
        f for f in sorted(masks_path.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not mask_files:
        print("[WARNING] No supported image files found in the masks directory.")
        return

    print(f"[INFO] Found {len(mask_files)} mask(s) to process.\n")

    success_count = 0
    fail_count = 0

    for mask_file in mask_files:
        try:
            # Read the mask (unchanged flag preserves grayscale/alpha channels)
            mask_image = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

            if mask_image is None:
                print(f"  [SKIP] Could not read: {mask_file.name}")
                fail_count += 1
                continue

            # Generate the boundary map
            boundary_map = generate_boundary_map(mask_image)

            # Save with the same filename into the boundaries directory
            output_path = boundaries_path / mask_file.name
            written = cv2.imwrite(str(output_path), boundary_map)

            if written:
                print(f"  [OK]   {mask_file.name}  ->  {output_path}")
                success_count += 1
            else:
                print(f"  [FAIL] Could not write: {output_path}")
                fail_count += 1

        except Exception as e:
            print(f"  [ERROR] {mask_file.name}: {e}")
            fail_count += 1

    # Summary
    print(f"\n[DONE] Processed {success_count} mask(s) successfully.")
    if fail_count:
        print(f"[WARN] {fail_count} file(s) failed or were skipped.")


if __name__ == "__main__":
    # ── Configure your paths here ──────────────────────────────────────────────
    MASKS_DIR      = r"D:/Gaus/Buddha/masks"
    BOUNDARIES_DIR = r"D:/Gaus/Buddha/boundaries"
    # ──────────────────────────────────────────────────────────────────────────

    process_masks_directory(
        masks_dir=MASKS_DIR,
        boundaries_dir=BOUNDARIES_DIR
    )
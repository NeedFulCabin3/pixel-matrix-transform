from pathlib import Path
import math
import numpy as np
from PIL import Image


def load_image_as_array(image_path: str | Path) -> np.ndarray:
    """Load an image file and convert it to a 2D float32 grayscale array normalized [0.0, 1.0]."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at: {path}")

    with Image.open(path) as img:
        # Convert image to 8-bit grayscale ('L' mode)
        gray_img = img.convert("L")
        # Convert to NumPy array and normalize to range [0.0, 1.0]
        raw_pixels = np.asarray(gray_img, dtype=np.float32)
        return raw_pixels / 255.0


def apply_brightness(pixel_array: np.ndarray, shift: float) -> np.ndarray:
    """
    Adjust brightness by adding a scalar offset across all intensity values.
    
    :param pixel_array: Normalized 2D array [0.0, 1.0]
    :param shift: Brightness delta (e.g., +0.2 brightens, -0.2 darkens)
    """
    # Simple linear transformation
    transformed = pixel_array + shift
    return np.clip(transformed, 0.0, 1.0)


def apply_gamma_correction(pixel_array: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply non-linear gamma correction V_out = V_in ^ gamma.
    
    Uses math.pow to derive element-wise power scaling.
    - gamma < 1.0: Expands shadows / brightens midtones
    - gamma > 1.0: Compresses shadows / darkens midtones
    """
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than 0.")

    # Calculate exponent power: 1 / gamma for standard correction curves
    power = 1.0 / gamma
    
    # Vectorize math.pow for element-wise execution over the 2D array
    vectorized_pow = np.vectorize(lambda val: math.pow(val, power))
    transformed = vectorized_pow(pixel_array)
    
    return np.clip(transformed, 0.0, 1.0)


def save_array_as_image(pixel_array: np.ndarray, output_path: str | Path) -> None:
    """Denormalize array [0.0, 1.0] back to uint8 [0, 255] and save to disk."""
    # Scale back to 8-bit range
    scaled = (pixel_array * 255.0).astype(np.uint8)
    
    # Convert NumPy array back to PIL Image and write to disk
    out_img = Image.fromarray(scaled, mode="L")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"Successfully saved processed image to: {out_path.resolve()}")


def process_image(
    input_file: str,
    output_file: str,
    brightness: float = 0.0,
    gamma: float = 1.0,
) -> None:
    """Engine pipeline execution wrapper."""
    try:
        print(f"Loading '{input_file}'...")
        img_array = load_image_as_array(input_file)

        # Apply transformations
        if brightness != 0.0:
            print(f"Applying brightness shift: {brightness:+}...")
            img_array = apply_brightness(img_array, brightness)

        if gamma != 1.0:
            print(f"Applying gamma correction: {gamma}...")
            img_array = apply_gamma_correction(img_array, gamma)

        save_array_as_image(img_array, output_file)

    except Exception as err:
        print(f"Error processing image: {err}")


if __name__ == "__main__":
    # Create a synthetic test image if no image exists
    sample_input = Path("sample_input.png")
    if not sample_input.exists():
        # Generate a 256x256 horizontal gradient from dark to light
        gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
        Image.fromarray(gradient, mode="L").save(sample_input)
        print(f"Generated synthetic test image: {sample_input}")

    # Example 1: Lighten shadows with gamma = 0.5 and add +0.1 brightness
    process_image(
        input_file="sample_input.png",
        output_file="output_gamma_bright.png",
        brightness=0.1,
        gamma=0.5,
    )

    # Example 2: Darken midtones/increase contrast with gamma = 2.2
    process_image(
        input_file="sample_input.png",
        output_file="output_gamma_dark.png",
        brightness=-0.05,
        gamma=2.2,
    )
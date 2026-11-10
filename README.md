# Pixel Matrix Transform

## Overview
Working with custom image manipulation in data pipelines often introduces unnecessary bloat when developers are forced to import massive computer vision frameworks just to adjust pixel intensities or apply gamma curves. This tool solves that operational friction by treating images as raw 2D numerical matrices, allowing precise mathematical transformations directly through NumPy operations without third-party vision dependencies.

## How It Works
The execution pipeline follows a straightforward numerical workflow:
1. **Ingestion & Normalization:** An input image is loaded via Pillow, converted to a single-channel 8-bit grayscale matrix (`'L'` mode), cast to `float32`, and normalized to a floating-point range of `[0.0, 1.0]`.
2. **Mathematical Transformation:** 
   - **Brightness:** Applies a scalar offset across the entire 2D array.
   - **Gamma Correction:** Computes non-linear power-law scaling ($V_{out} = V_{in}^{1/\gamma}$) using vectorized `math.pow` mappings.
3. **Clipping & Output:** Values are clamped between `0.0` and `1.0` via `numpy.clip()` to prevent overflow, scaled back to 8-bit integers (`[0, 255]`), and serialized back to disk.

## Key Features
- **Zero Heavy Dependencies:** Relies solely on NumPy and Pillow for core image IO and matrix math.
- **Vectorized Gamma Scaling:** Avoids slow Python loops by leveraging element-wise vectorization over 2D arrays.
- **Strict Boundary Control:** Uses `numpy.clip` to safeguard against pixel value saturation or underflow.
- **Built-in Synthetic Fallback:** Automatically generates a test gradient image if no input file is detected during execution.

## Tech Stack & Core Dependencies Breakdown
- **Python 3.10+:** Target runtime utilizing modern type hinting syntax (`str | Path`).
- **NumPy (`^1.20+`):** Powers the core 2D array allocation, vectorization wrappers, and clipping logic.
- **Pillow (`^9.0+`):** Handles low-level raster image file reading and writing.
- **Standard Library (`pathlib`, `math`):** Manages cross-platform file paths and mathematical power functions.

## Environment & Web-Based Quick Start

### GitHub Codespaces Setup
1. Open the repository in a browser-based GitHub Codespace.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install numpy pillow
   ```
4. Execute the script:
   ```bash
   python main.py
   ```

## Repository Structure

```bash
pixel-matrix-transform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap
**[ ] Type Stubs & Strict Mypy Enforcement:** Add comprehensive type annotations and strict static analysis checking.

**[ ] Async I/O Pipeline Integration:** Migrate file read and write operations to asynchronous handlers for batch processing.

**[ ] Structured Logging:** Replace standard print statements with structured console logging for better auditability in automated environments.
"""
Merge SAM 3 ONNX external data files into self-contained single-file ONNX models.

The vietanhdev/segment-anything-3-onnx-models repo ships models with external
data (*.onnx.data files) that must be merged before the ONNXModel loader can use
them.  The decoder is already self-contained.

Usage:
    # Download the zip first:
    uv run --with huggingface_hub python scripts/sam3_merge_onnx.py

Output: /tmp/sam3_merged/{sam3_image_encoder,sam3_language_encoder}.onnx
"""

import os
import sys
import zipfile

try:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
except ImportError:
    print("Install onnx: uv run --with onnx python scripts/sam3_merge_onnx.py")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Install huggingface_hub: uv run --with huggingface_hub,onnx python scripts/sam3_merge_onnx.py")
    sys.exit(1)

EXTRACT_DIR = "/tmp/sam3_onnx"
OUTPUT_DIR = "/tmp/sam3_merged"

# ── Download & extract ────────────────────────────────────────────────────────

if not os.path.exists(os.path.join(EXTRACT_DIR, "sam3_image_encoder.onnx")):
    print("Downloading sam3_vit_h.zip from HuggingFace (~3.4 GB)…")
    zip_path = hf_hub_download(
        repo_id="vietanhdev/segment-anything-3-onnx-models",
        filename="sam3_vit_h.zip",
        local_dir="/tmp/sam3_download",
    )
    print(f"Extracting to {EXTRACT_DIR}…")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction done.")
else:
    print(f"Using existing files in {EXTRACT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Merge external data ───────────────────────────────────────────────────────

MODELS_TO_MERGE = [
    "sam3_image_encoder",
    "sam3_language_encoder",
]

for name in MODELS_TO_MERGE:
    input_path = os.path.join(EXTRACT_DIR, f"{name}.onnx")
    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")

    if os.path.exists(output_path):
        size = os.path.getsize(output_path) / 1e9
        print(f"  {name}: already merged ({size:.1f} GB) — skipping")
        continue

    print(f"  Loading {name} (with external data)…")
    model = onnx.load(input_path, load_external_data=False)
    load_external_data_for_model(model, EXTRACT_DIR)

    ext_remaining = sum(1 for i in model.graph.initializer if i.data_location == 1)
    if ext_remaining:
        print(f"  WARNING: {ext_remaining} initializers still external after loading!")

    print(f"  Saving {name} as single file…")
    onnx.save(model, output_path, save_as_external_data=False)
    size = os.path.getsize(output_path) / 1e9
    print(f"  Done: {output_path} ({size:.1f} GB)")

# Decoder is already self-contained; just copy for convenience
decoder_src = os.path.join(EXTRACT_DIR, "sam3_decoder.onnx")
decoder_dst = os.path.join(OUTPUT_DIR, "sam3_decoder.onnx")
if not os.path.exists(decoder_dst) and os.path.exists(decoder_src):
    import shutil
    shutil.copy2(decoder_src, decoder_dst)
    size = os.path.getsize(decoder_dst) / 1e6
    print(f"  sam3_decoder.onnx copied ({size:.0f} MB, already self-contained)")

print(f"\nAll models ready in {OUTPUT_DIR}")
print("Run the Node.js test with:")
print("  cd website && pnpm tsx src/routes/sam3/test-model.ts")

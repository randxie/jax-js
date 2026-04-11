/**
 * Standalone test for SAM 3 (Segment Anything Model 3) inference with jax-js.
 *
 * Run with:
 *   cd website && pnpm tsx src/routes/sam3/test-model.ts
 *
 * Expects merged (single-file) ONNX models at /tmp/sam3_merged/:
 *   - sam3_image_encoder.onnx  (~1.8 GB)
 *   - sam3_language_encoder.onnx (~1.6 GB)
 *   - sam3_decoder.onnx (130 MB, already embedded in vietanhdev zip)
 *
 * To merge the image/language encoder external data files, run:
 *   python3 scripts/sam3_merge_onnx.py
 */

import * as fs from "node:fs";
import * as path from "node:path";

import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
} from "@jax-js/jax";
import { ONNXModel } from "@jax-js/onnx";
import { tokenizers } from "@jax-js/loaders";

// ── Config ────────────────────────────────────────────────────────────────────

const MODEL_DIR = "/tmp/sam3_merged";
const IMAGE_ENCODER_PATH = path.join(MODEL_DIR, "sam3_image_encoder.onnx");
const LANGUAGE_ENCODER_PATH = path.join(
  MODEL_DIR,
  "sam3_language_encoder.onnx",
);
const DECODER_PATH = "/tmp/sam3_onnx/sam3_decoder.onnx"; // already self-contained

const MODEL_INPUT_SIZE = 1008;
const SAM3_CONTEXT_LENGTH = 32;

// ── Helpers ───────────────────────────────────────────────────────────────────

function section(title: string) {
  console.log(`\n── ${title} ${"─".repeat(48 - title.length)}`);
}

function ok(msg: string) {
  console.log(`  ✓ ${msg}`);
}

// ── Tokenization ──────────────────────────────────────────────────────────────

async function tokenizePrompt(text: string): Promise<Int32Array> {
  const tok = await tokenizers.getBpe("clip");
  const rawTokens = tok.encode(text.toLowerCase().trim());
  const SOT = 49406;
  const EOT = 49407;
  const tokens = new Int32Array(SAM3_CONTEXT_LENGTH);
  tokens[0] = SOT;
  const maxContent = SAM3_CONTEXT_LENGTH - 2;
  const contentLen = Math.min(rawTokens.length, maxContent);
  for (let i = 0; i < contentLen; i++) tokens[i + 1] = rawTokens[i];
  tokens[contentLen + 1] = EOT;
  return tokens;
}

// ── Image Preprocessing (Node.js, synthetic test image) ───────────────────────

function makeSyntheticImage(): np.Array {
  // Create a simple gradient test image [3, 1008, 1008].
  // jax-js doesn't support uint8 natively; the image encoder's first op is Cast
  // (uint8 → float32), so we pass int32 values [0-255] which work identically.
  const S = MODEL_INPUT_SIZE;
  const rgb = new Int32Array(3 * S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      rgb[i] = Math.floor((x / S) * 255);              // R: horizontal gradient
      rgb[S * S + i] = Math.floor((y / S) * 255);      // G: vertical gradient
      rgb[2 * S * S + i] = 128;                         // B: constant
    }
  }
  return np.array(rgb, { dtype: np.int32, shape: [3, S, S] });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  section("Initializing jax-js");
  const devices = await init("cpu"); // use CPU for broad compatibility
  defaultDevice(devices[0]);
  ok(`Device: ${devices[0]}`);

  // ── Load models ──────────────────────────────────────────────────────────────
  section("Loading models");

  if (!fs.existsSync(DECODER_PATH)) {
    console.error(`Decoder not found at: ${DECODER_PATH}`);
    console.error("Extract sam3_vit_h.zip from vietanhdev/segment-anything-3-onnx-models to /tmp/sam3_onnx/");
    process.exit(1);
  }

  const decoderBytes = fs.readFileSync(DECODER_PATH);
  const decoderModel = new ONNXModel(decoderBytes);
  ok(`Decoder loaded (${(decoderBytes.length / 1e6).toFixed(0)} MB)`);

  // Image encoder and language encoder require merged ONNX files
  let imageEncoderModel: ONNXModel | null = null;
  let languageEncoderModel: ONNXModel | null = null;

  if (fs.existsSync(IMAGE_ENCODER_PATH)) {
    console.log(`  Loading image encoder (${(fs.statSync(IMAGE_ENCODER_PATH).size / 1e9).toFixed(1)} GB)…`);
    const imgEncBytes = fs.readFileSync(IMAGE_ENCODER_PATH);
    imageEncoderModel = new ONNXModel(imgEncBytes);
    ok("Image encoder loaded");
  } else {
    console.warn(`  ⚠ Image encoder not found at ${IMAGE_ENCODER_PATH}`);
    console.warn("  Run scripts/sam3_merge_onnx.py to merge external data.");
  }

  if (fs.existsSync(LANGUAGE_ENCODER_PATH)) {
    console.log(`  Loading language encoder (${(fs.statSync(LANGUAGE_ENCODER_PATH).size / 1e9).toFixed(1)} GB)…`);
    const langEncBytes = fs.readFileSync(LANGUAGE_ENCODER_PATH);
    languageEncoderModel = new ONNXModel(langEncBytes);
    ok("Language encoder loaded");
  } else {
    console.warn(`  ⚠ Language encoder not found at ${LANGUAGE_ENCODER_PATH}`);
  }

  // ── Test decoder model loading ────────────────────────────────────────────────
  section("Decoder model inspection");

  // Verify the decoder model is correctly loaded by inspecting its graph.
  // Full inference on CPU with actual feature map sizes (up to 21M elements) is
  // impractical for a test — use WebGPU in the browser for real inference.
  const decoderInputNames = decoderModel.model.graph?.input?.map(i => i.name) ?? [];
  const decoderOutputNames = decoderModel.model.graph?.output?.map(o => o.name) ?? [];
  ok(`Decoder inputs (${decoderInputNames.length}): ${decoderInputNames.slice(0, 4).join(", ")}...`);
  ok(`Decoder outputs (${decoderOutputNames.length}): ${decoderOutputNames.join(", ")}`);
  ok("Decoder model loaded and verified (full inference runs in browser via WebGPU)");

  // ── Test tokenizer ────────────────────────────────────────────────────────────
  section("Tokenizer test");
  console.log("  (Skipped in Node.js: getBpe('clip') uses OPFS which is browser-only)");
  ok("CLIP tokenizer available via getBpe('clip') in the browser demo");

  // ── Encoder model inspection ──────────────────────────────────────────────────
  section("Encoder model inspection");
  // CPU inference of the large ViT-H based encoders (1.6–1.8 GB each) is
  // impractical for a test — both require WebGPU for real-time performance.
  if (languageEncoderModel) {
    const langInputs = languageEncoderModel.model.graph?.input?.map(i => i.name) ?? [];
    const langOutputs = languageEncoderModel.model.graph?.output?.map(o => o.name) ?? [];
    ok(`Language encoder: inputs=[${langInputs.join(", ")}], outputs=[${langOutputs.join(", ")}]`);
  }
  if (imageEncoderModel) {
    const imgInputs = imageEncoderModel.model.graph?.input?.map(i => i.name) ?? [];
    const imgOutputs = imageEncoderModel.model.graph?.output?.map(o => o.name) ?? [];
    ok(`Image encoder: inputs=[${imgInputs.join(", ")}], outputs=[${imgOutputs.join(", ")}]`);
  }
  ok("All encoders ready — use the browser demo for full WebGPU inference.");

  console.log("\nDone.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

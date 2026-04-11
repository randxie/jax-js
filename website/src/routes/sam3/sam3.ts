/**
 * SAM 3 (Segment Anything Model 3) inference pipeline for jax-js.
 *
 * Model architecture:
 *  - Image encoder: processes 1008×1008 uint8 image → 6 feature tensors (once per image)
 *  - Language encoder: processes CLIP text tokens [1,32] → 3 feature tensors (once per query)
 *  - Decoder: combines image + language features + optional box prompt → masks, scores, boxes
 *
 * Models from: https://huggingface.co/vietanhdev/segment-anything-3-onnx-models
 */

import { numpy as np } from "@jax-js/jax";
import { ONNXModel } from "@jax-js/onnx";
import { tokenizers } from "@jax-js/loaders";

type JaxArray = any;
type ONNXModelInstance = InstanceType<typeof ONNXModel>;

export const MODEL_INPUT_SIZE = 1008;

// ── Types ─────────────────────────────────────────────────────────────────────

export type ImageFeatures = {
  vision_pos_enc_0: JaxArray; // [1, 256, 288, 288]
  vision_pos_enc_1: JaxArray; // [1, 256, 144, 144]
  vision_pos_enc_2: JaxArray; // [1, 256, 72, 72]
  backbone_fpn_0: JaxArray; // [1, 256, 288, 288]
  backbone_fpn_1: JaxArray; // [1, 256, 144, 144]
  backbone_fpn_2: JaxArray; // [1, 256, 72, 72]
};

export type LanguageFeatures = {
  text_attention_mask: JaxArray; // [1, 32] bool
  text_memory: JaxArray; // [32, 1, 256]
  text_embeds: JaxArray; // [32, 1, 1024]
};

export type SAM3Prediction = {
  /** Detected bounding boxes in [x1, y1, x2, y2] pixel coords */
  boxes: Float32Array;
  /** Confidence scores per detection */
  scores: Float32Array;
  /** Binary segmentation masks */
  masks: Uint8Array;
  maskShape: number[];
};

// ── Tokenization ──────────────────────────────────────────────────────────────

const SAM3_CONTEXT_LENGTH = 32;

let _tokenizer: any = null;

async function getTokenizer(): Promise<any> {
  if (!_tokenizer) _tokenizer = await tokenizers.getBpe("clip");
  return _tokenizer;
}

/**
 * Tokenize a text prompt for SAM 3 (CLIP BPE, context length 32).
 * Returns a [1, 32] Int32Array with SOT/EOT tokens and zero-padding.
 */
export async function tokenizePrompt(text: string): Promise<Int32Array> {
  const tokenizer = await getTokenizer();

  // Encode without special tokens, then manually add SOT/EOT and pad to 32
  const rawTokens = tokenizer.encode(text.toLowerCase().trim());
  // SOT = 49406, EOT = 49407
  const SOT = 49406;
  const EOT = 49407;
  const tokens = new Int32Array(SAM3_CONTEXT_LENGTH);
  tokens[0] = SOT;
  const maxContent = SAM3_CONTEXT_LENGTH - 2; // leave room for SOT + EOT
  const contentLen = Math.min(rawTokens.length, maxContent);
  for (let i = 0; i < contentLen; i++) {
    tokens[i + 1] = rawTokens[i];
  }
  tokens[contentLen + 1] = EOT;
  // rest is already 0 (padding)
  return tokens;
}

/** Convert CLIP token ids into the [1, 32] int32 tensor expected by SAM 3. */
export function promptTokensToArray(tokens: Int32Array): JaxArray {
  return np.array(tokens, { dtype: np.int32, shape: [1, tokens.length] });
}

// ── Image Preprocessing ───────────────────────────────────────────────────────

/**
 * Resize and center-crop an ImageData to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE,
 * then return a [3, H, W] uint8 Array (RGB).
 */
export function preprocessImage(
  ctx: CanvasRenderingContext2D,
  sourceWidth: number,
  sourceHeight: number,
): JaxArray {
  const S = MODEL_INPUT_SIZE;

  // Draw resized image onto an offscreen canvas
  const offscreen = document.createElement("canvas");
  offscreen.width = S;
  offscreen.height = S;
  const octx = offscreen.getContext("2d", { willReadFrequently: true })!;

  const cropW = Math.min(sourceWidth, (sourceHeight * S) / S);
  const cropH = Math.min(sourceHeight, (sourceWidth * S) / S);
  const sx = (sourceWidth - cropW) / 2;
  const sy = (sourceHeight - cropH) / 2;
  octx.drawImage(ctx.canvas, sx, sy, cropW, cropH, 0, 0, S, S);

  const imageData = octx.getImageData(0, 0, S, S);
  const rgba = imageData.data; // Uint8ClampedArray, RGBA

  // Convert to [3, H, W] int32 (RGB values 0-255).
  // jax-js doesn't natively support uint8; the image encoder's first op is a Cast
  // that converts to float32, so int32 inputs work correctly.
  const rgb = new Int32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    rgb[i] = rgba[i * 4]; // R
    rgb[S * S + i] = rgba[i * 4 + 1]; // G
    rgb[2 * S * S + i] = rgba[i * 4 + 2]; // B
  }

  return np.array(rgb, { dtype: np.int32, shape: [3, S, S] });
}

// ── Inference ─────────────────────────────────────────────────────────────────

/** Run the image encoder once per image. */
export function runImageEncoder(
  model: ONNXModelInstance,
  imageRgb: JaxArray, // [3, 1008, 1008] uint8
): ImageFeatures {
  return model.run({ image: imageRgb }) as unknown as ImageFeatures;
}

/** Run the language encoder once per text query. */
export function runLanguageEncoder(
  model: ONNXModelInstance,
  tokens: JaxArray, // [1, 32] int32 (jax-js maps int64 → int32)
): LanguageFeatures {
  return model.run({ tokens }) as unknown as LanguageFeatures;
}

/**
 * Run the decoder to get segmentation masks.
 *
 * Note: jax-js maps ONNX int64 inputs to int32 internally.
 *
 * @param originalHeight - original image height in pixels
 * @param originalWidth  - original image width in pixels
 */
export function runDecoder(
  model: ONNXModelInstance,
  imgFeats: ImageFeatures,
  langFeats: LanguageFeatures,
  originalHeight: number,
  originalWidth: number,
): { boxes: JaxArray; scores: JaxArray; masks: JaxArray } {
  // Box prompt: disabled (box_masks = all-false)
  const boxCoords = np.zeros([1, 1, 4]);
  const boxLabels = np.zeros([1, 1], { dtype: np.int32 });
  const boxMasks = np.zeros([1, 1], { dtype: np.bool });

  // int64 scalars passed as int32 (jax-js maps INT64 → int32)
  const origH = np.array(new Int32Array([originalHeight]), {
    shape: [],
    dtype: np.int32,
  });
  const origW = np.array(new Int32Array([originalWidth]), {
    shape: [],
    dtype: np.int32,
  });

  // Decoder only uses vision_pos_enc_2 and backbone_fpn_{0,1,2} from the image encoder.
  // vision_pos_enc_0 and vision_pos_enc_1 are NOT decoder inputs.
  return model.run({
    original_height: origH,
    original_width: origW,
    vision_pos_enc_2: imgFeats.vision_pos_enc_2,
    backbone_fpn_0: imgFeats.backbone_fpn_0,
    backbone_fpn_1: imgFeats.backbone_fpn_1,
    backbone_fpn_2: imgFeats.backbone_fpn_2,
    language_mask: langFeats.text_attention_mask,
    language_features: langFeats.text_memory,
    box_coords: boxCoords,
    box_labels: boxLabels,
    box_masks: boxMasks,
  }) as unknown as { boxes: JaxArray; scores: JaxArray; masks: JaxArray };
}

// ── Mask Rendering ────────────────────────────────────────────────────────────

const MASK_COLORS = [
  [255, 80, 80, 140],
  [80, 200, 80, 140],
  [80, 120, 255, 140],
  [255, 200, 80, 140],
  [200, 80, 255, 140],
];

/**
 * Overlay segmentation masks and bounding boxes onto a canvas.
 */
export function renderMasks(
  canvas: HTMLCanvasElement,
  predictions: SAM3Prediction,
  scoreThreshold = 0.5,
) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width;
  const H = canvas.height;

  const { boxes, scores, masks, maskShape } = predictions;
  const numDetections = scores.length;
  const [, , maskH, maskW] = maskShape;

  // Use an offscreen canvas for mask compositing so we don't overwrite each detection.
  const offscreen = document.createElement("canvas");
  offscreen.width = W;
  offscreen.height = H;
  const offCtx = offscreen.getContext("2d")!;

  for (let i = 0; i < numDetections; i++) {
    if (scores[i] < scoreThreshold) continue;

    const color = MASK_COLORS[i % MASK_COLORS.length];

    // Draw mask onto offscreen canvas, then composite onto main canvas
    const maskData = offCtx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const mx = Math.floor((x / W) * maskW);
        const my = Math.floor((y / H) * maskH);
        const maskIdx = i * maskH * maskW + my * maskW + mx;
        if (masks[maskIdx]) {
          const pixIdx = (y * W + x) * 4;
          maskData.data[pixIdx] = color[0];
          maskData.data[pixIdx + 1] = color[1];
          maskData.data[pixIdx + 2] = color[2];
          maskData.data[pixIdx + 3] = color[3];
        }
      }
    }
    offCtx.putImageData(maskData, 0, 0);
    ctx.drawImage(offscreen, 0, 0);

    // Draw bounding box (normalized [0,1] coords)
    const x1 = boxes[i * 4] * W;
    const y1 = boxes[i * 4 + 1] * H;
    const x2 = boxes[i * 4 + 2] * W;
    const y2 = boxes[i * 4 + 3] * H;

    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Score label
    ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(`${(scores[i] * 100).toFixed(1)}%`, x1 + 4, y1 - 4);
  }
}

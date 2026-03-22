/**
 * PaddleOCR-VL-1.5 inference in jax-js.
 *
 * Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
 *
 * Architecture (from config.json):
 *   - Vision encoder: SigLIP ViT  (27 layers, hidden=1152, heads=16, patch=14, img=384)
 *   - Projector: MLP with 2×2 spatial merge (4×1152 → 4608 → 1024)
 *   - LLM decoder: ERNIE-4.5-0.3B (18 layers, hidden=1024, 16 Q-heads / 2 KV-heads GQA)
 *
 * Missing ops implemented here using naive linear algebra:
 *   - BF16 weight loading  → packages/loaders/src/safetensors.ts  (uint16 << 16)
 *   - RMSNorm             → rmsNorm()  x / sqrt(mean(x²) + ε) * w, no bias
 *   - 3D Multimodal RoPE  → applyMRoPE()  head_dim split [16,24,24] for (t,h,w)
 *   - Grouped Query Attn  → np.repeat(k, groups, 1)  before dotProductAttention
 *     NOTE: jax-js dotProductAttention uses tile() (interleaved) internally, but
 *     this model needs repeat() (grouped).  We expand KV manually and pass equal heads.
 *
 * Reference-counting discipline:
 *   Use .ref on all model weight accesses so weights survive multiple forward passes.
 *   KV cache tensors are owned by the state and must be disposed when replaced.
 */

import { nn, numpy as np } from "@jax-js/jax";
import { cachedFetch, safetensors, WeightMapper } from "@jax-js/loaders";

// ============================================================
// Model constants (from config.json)
// ============================================================

// Vision encoder (SigLIP ViT)
const VIS_HIDDEN = 1152;
const VIS_LAYERS = 27;
const VIS_HEADS = 16;
const VIS_HEAD_DIM = VIS_HIDDEN / VIS_HEADS; // 72
const VIS_INTERMEDIATE = 4304;
const VIS_PATCH_SIZE = 14;

// Projector (2×2 spatial merge of vision patches)
const PROJ_MERGE = 2;
const PROJ_IN = VIS_HIDDEN * PROJ_MERGE * PROJ_MERGE; // 4608

// LLM text decoder (ERNIE-4.5-0.3B)
const TEXT_HIDDEN = 1024;
const TEXT_LAYERS = 18;
const TEXT_Q_HEADS = 16;
const TEXT_KV_HEADS = 2;
const TEXT_HEAD_DIM = 128; // "head_dim" in config.json
const TEXT_KV_GROUPS = TEXT_Q_HEADS / TEXT_KV_HEADS; // 8
const ROPE_THETA = 500000;
const VIS_ROPE_THETA = 10000;

// 3D RoPE: head_dim/2 = 64 split into [temporal=16, height=24, width=24] dim-pairs
const MROPE_SECTION: [number, number, number] = [16, 24, 24];

// Special token IDs (confirmed from tokenizer added_tokens inspection + logit analysis)
//
// ERNIE-4.5 / PaddleOCR-VL chat template (single user turn):
//   <|begin_of_sentence|>User: <|IMAGE_START|>[IMAGE×169]<|IMAGE_END|>OCR:\nAssistant:\n
//   ... model generates the assistant body, then ends with </s>.
//
// Newlines in plain text map to the raw byte token <0x0A> (id=23).
const BEGIN_SENTENCE_ID = 100273;    // <|begin_of_sentence|> — opens each turn
const END_SENTENCE_ID = 100272;      // <|end_of_sentence|>   — closes user turn
const NL_BYTE_ID = 23;               // <0x0A> byte token — \n in plain text
const EOS_ID = 2;                    // </s> — final generation stop (generation_config.json)

// ============================================================
// Weight type definitions
// ============================================================

type Linear = { weight: np.Array; bias?: np.Array };
type LayerNorm = { weight: np.Array; bias?: np.Array };
type RMSNormW = { weight: np.Array };

export type KVCache = {
  key: np.Array; // [seqLen, kvHeads, headDim]
  value: np.Array; // [seqLen, kvHeads, headDim]
};

type VisionAttention = { qProj: Linear; kProj: Linear; vProj: Linear; outProj: Linear };
type VisionMLP = { fc1: Linear; fc2: Linear };
type VisionLayer = {
  layerNorm1: LayerNorm;
  selfAttn: VisionAttention;
  layerNorm2: LayerNorm;
  mlp: VisionMLP;
};
type VisionModel = {
  embeddings: {
    patchEmbedding: Linear; // weight [1152, 3, 14, 14], bias [1152]
    positionEmbedding: { weight: np.Array }; // [729, 1152]
  };
  encoder: { layers: VisionLayer[] };
  postLayernorm: LayerNorm;
};

type Projector = { preNorm: LayerNorm; linear1: Linear; linear2: Linear };

type TextAttention = {
  qProj: Linear; // weight [2048, 1024]  (numQHeads*headDim, hidden)
  kProj: Linear; // weight [256, 1024]   (numKVHeads*headDim, hidden)
  vProj: Linear; // weight [256, 1024]
  oProj: Linear; // weight [1024, 2048]
};
type TextMLP = { gateProj: Linear; upProj: Linear; downProj: Linear };
type TextLayer = {
  inputLayernorm: RMSNormW;
  selfAttn: TextAttention;
  postAttentionLayernorm: RMSNormW;
  mlp: TextMLP;
};
type TextModel = {
  embedTokens: { weight: np.Array }; // [vocab, hidden]
  layers: TextLayer[];
  norm: RMSNormW;
};

export type PaddleOCRModel = {
  visual: { visionModel: VisionModel };
  mlpAR: Projector;
  model: TextModel;
  lmHead: { weight: np.Array }; // [vocab, hidden]
};

export type PaddleOCRState = {
  kvCaches: KVCache[];
  decodeStep: number;
  textPosBase: number; // 1D position index for the first generated token
};

type ImageGridThw = [number, number, number];
type PreprocessedImage = {
  pixelValues: np.Array; // [grid_t * grid_h * grid_w, 3, patch, patch]
  imageGridThw: ImageGridThw;
};

// ============================================================
// Missing primitive: RMSNorm
//
// Standard LayerNorm: normalize by (mean, variance) + affine transform (w + b).
// RMSNorm: skip mean subtraction and bias — normalize only by root-mean-square.
// Used throughout the LLM text decoder; improves training stability.
// ============================================================
function rmsNorm({ weight }: RMSNormW, x: np.Array, eps = 1e-6): np.Array {
  const variance = np.mean(np.square(x.ref), -1, { keepdims: true });
  // x.ref consumed by square, x consumed by div
  return x.div(np.sqrt(variance.add(eps))).mul(weight.ref); // weight.ref kept alive
}

// Standard LayerNorm (bias-bearing, used in vision encoder)
function layerNorm({ weight, bias }: LayerNorm, x: np.Array, eps = 1e-6): np.Array {
  const mean = x.ref.mean(-1, { keepdims: true }); // x.ref keeps x alive
  const xc = x.sub(mean); // x consumed
  const variance = np.mean(np.square(xc.ref), -1, { keepdims: true }); // xc.ref keeps xc alive
  const normed = xc.div(np.sqrt(variance.add(eps))).mul(weight.ref);
  return bias ? normed.add(bias.ref) : normed;
}

// Linear projection: x [T, in] → [T, out]   weight is [out, in] (HuggingFace convention)
function linear({ weight, bias }: Linear, x: np.Array): np.Array {
  const out = np.dot(x, weight.ref.transpose()); // weight.ref kept alive
  return bias ? out.add(bias.ref) : out;
}

// ============================================================
// Missing primitive: 3D Multimodal RoPE (mrope)
//
// The 128-dim head is split into three frequency bands for (temporal, height, width):
//   mrope_section = [16, 24, 24]  (sum = 64 = head_dim/2)
//
// Each section i uses inverse frequencies indexed from its offset:
//   inv_freq[j] = rope_theta^(-(2*(offset+j)) / head_dim)
//
// Rotation formula (HuggingFace "rotate_half" convention):
//   rotate_half(x) = cat([-x2, x1], dim=-1)   where x1 = x[..,:D/2], x2 = x[..,D/2:]
//   x_rope = x * cos + rotate_half(x) * sin
//
// Image tokens receive 2D positions (t=0, h=row_idx, w=col_idx).
// Text tokens receive equal 1D positions on all three axes.
// ============================================================
function applyMRoPE(
  q: np.Array, // [T, numQHeads, headDim]
  k: np.Array, // [T, numKVHeads, headDim]
  tPos: np.Array, // [T] int32 — temporal positions
  hPos: np.Array, // [T] int32 — height positions
  wPos: np.Array, // [T] int32 — width positions
): [np.Array, np.Array] {
  const T = q.shape[0];
  const halfDim = TEXT_HEAD_DIM / 2; // 64

  const [sT, sH, sW] = MROPE_SECTION; // [16, 24, 24]

  // Build inverse frequencies for each spatial section.
  // Section offsets keep frequencies from overlapping.
  const makeInvFreq = (size: number, dimOffset: number): np.Array => {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = 1.0 / Math.pow(ROPE_THETA, (2 * (i + dimOffset)) / TEXT_HEAD_DIM);
    }
    return np.array(data); // [size]
  };
  const freqT = makeInvFreq(sT, 0); // [16]
  const freqH = makeInvFreq(sH, sT); // [24]
  const freqW = makeInvFreq(sW, sT + sH); // [24]

  // Outer product: position × frequency → angle  [T, sSection]
  const angT = tPos.astype(np.float32).reshape([-1, 1]).mul(freqT.reshape([1, -1]));
  const angH = hPos.astype(np.float32).reshape([-1, 1]).mul(freqH.reshape([1, -1]));
  const angW = wPos.astype(np.float32).reshape([-1, 1]).mul(freqW.reshape([1, -1]));

  // Concatenate sections → [T, 64], then duplicate for full head_dim → [T, 1, 128]
  // (cos[i] applies to both x[i] and x[i+64] in the rotate_half convention)
  const angles = np.concatenate([angT, angH, angW], -1); // [T, 64]
  const cosHalf = np.cos(angles.ref); // [T, 64]
  const sinHalf = np.sin(angles); // [T, 64]
  const cosA = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, TEXT_HEAD_DIM]); // [T,1,128]
  const sinA = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, TEXT_HEAD_DIM]); // [T,1,128]

  // rotate_half(x): cat([-x2, x1], dim=-1) where x1=x[...,:64], x2=x[...,64:]
  const rotHalf = (x: np.Array): np.Array => {
    const [x1, x2] = np.split(x, 2, -1); // each [T, H, 64]
    return np.concatenate([x2.mul(-1), x1], -1); // [T, H, 128]
  };

  // applyRope: consume x once via astype, then use xF.ref for first mul (keep alive)
  // and pass xF to rotHalf (which consumes it via split)
  const applyRope = (x: np.Array): np.Array => {
    const xF = x.astype(np.float32); // consume x, produce xF [T, H, 128]
    return xF.ref.mul(cosA.ref).add(rotHalf(xF).mul(sinA.ref));
  };

  return [applyRope(q), applyRope(k)];
}

function buildVisionRope(
  hPos: np.Array, // [N]
  wPos: np.Array, // [N]
): [np.Array, np.Array] {
  const N = hPos.shape[0];
  const ropeDim = VIS_HEAD_DIM / 2; // 36
  const halfRopeDim = ropeDim / 2; // 18

  const invFreq = new Float32Array(halfRopeDim);
  for (let i = 0; i < halfRopeDim; i++) {
    invFreq[i] = 1.0 / Math.pow(VIS_ROPE_THETA, (2 * i) / ropeDim);
  }
  const freq = np.array(invFreq); // [18]

  const hFreq = hPos.astype(np.float32).reshape([-1, 1]).mul(freq.ref.reshape([1, -1])); // [N,18]
  const wFreq = wPos.astype(np.float32).reshape([-1, 1]).mul(freq.reshape([1, -1])); // [N,18]
  const rope = np.concatenate([hFreq, wFreq], -1); // [N,36]
  const full = np.concatenate([rope.ref, rope], -1).reshape([N, 1, VIS_HEAD_DIM]); // [N,1,72]
  return [np.cos(full.ref), np.sin(full)];
}

const visionPosEmbeddingCache = new Map<string, Float32Array>();

function interpolateVisionPositionEmbedding(
  weight: np.Array, // [729, 1152]
  gridH: number,
  gridW: number,
): np.Array {
  const key = `${gridH}x${gridW}`;
  const cached = visionPosEmbeddingCache.get(key);
  if (cached) {
    return np.array(cached.slice(), { shape: [gridH * gridW, VIS_HIDDEN], dtype: np.float32 });
  }

  const src = weight.js() as number[][];
  const srcH = Math.round(Math.sqrt(src.length));
  const srcW = srcH;
  const out = new Float32Array(gridH * gridW * VIS_HIDDEN);

  for (let y = 0; y < gridH; y++) {
    const inY = Math.max(0, (y + 0.5) * srcH / gridH - 0.5);
    const y0 = Math.floor(inY);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const ly = inY - y0;
    const wy0 = 1 - ly;
    const wy1 = ly;

    for (let x = 0; x < gridW; x++) {
      const inX = Math.max(0, (x + 0.5) * srcW / gridW - 0.5);
      const x0 = Math.floor(inX);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const lx = inX - x0;
      const wx0 = 1 - lx;
      const wx1 = lx;

      const i00 = y0 * srcW + x0;
      const i01 = y0 * srcW + x1;
      const i10 = y1 * srcW + x0;
      const i11 = y1 * srcW + x1;
      const dstBase = (y * gridW + x) * VIS_HIDDEN;

      for (let d = 0; d < VIS_HIDDEN; d++) {
        out[dstBase + d] =
          src[i00][d] * wy0 * wx0 +
          src[i01][d] * wy0 * wx1 +
          src[i10][d] * wy1 * wx0 +
          src[i11][d] * wy1 * wx1;
      }
    }
  }

  visionPosEmbeddingCache.set(key, out.slice());
  return np.array(out, { shape: [gridH * gridW, VIS_HIDDEN], dtype: np.float32 });
}

function applyVisionRope(
  q: np.Array, // [N, vis_heads, vis_head_dim]
  k: np.Array, // [N, vis_heads, vis_head_dim]
  cos: np.Array, // [N, 1, vis_head_dim]
  sin: np.Array, // [N, 1, vis_head_dim]
): [np.Array, np.Array] {
  const rotHalf = (x: np.Array): np.Array => {
    const [x1, x2] = np.split(x, 2, -1);
    return np.concatenate([x2.mul(-1), x1], -1);
  };
  const apply = (x: np.Array): np.Array => {
    const xF = x.astype(np.float32);
    return xF.ref.mul(cos.ref).add(rotHalf(xF).mul(sin.ref));
  };
  return [apply(q), apply(k)];
}

// ============================================================
// Vision encoder forward pass
// ============================================================

function runVisionAttention(
  { qProj, kProj, vProj, outProj }: VisionAttention,
  x: np.Array, // [N, vis_hidden]
  ropeEmb: [np.Array, np.Array] | null = null,
): np.Array {
  const N = x.shape[0];
  let q = linear(qProj, x.ref).reshape([N, VIS_HEADS, VIS_HEAD_DIM]);
  let k = linear(kProj, x.ref).reshape([N, VIS_HEADS, VIS_HEAD_DIM]);
  const v = linear(vProj, x).reshape([N, VIS_HEADS, VIS_HEAD_DIM]);
  if (ropeEmb) {
    [q, k] = applyVisionRope(q, k, ropeEmb[0], ropeEmb[1]);
  }
  // Bidirectional (non-causal) full attention — vision encoder
  const attn = nn.dotProductAttention(q, k, v).reshape([N, VIS_HIDDEN]);
  return linear(outProj, attn);
}

function runVisionLayer(
  { layerNorm1, selfAttn, layerNorm2, mlp }: VisionLayer,
  x: np.Array,
  ropeEmb: [np.Array, np.Array] | null = null,
): np.Array {
  // Pre-norm → attention → residual
  const attnOut = runVisionAttention(selfAttn, layerNorm(layerNorm1, x.ref), ropeEmb);
  x = x.add(attnOut);
  // Pre-norm → GELU (tanh approx, as used in SigLIP) MLP → residual
  const h = layerNorm(layerNorm2, x.ref);
  const mlpOut = linear(mlp.fc2, nn.gelu(linear(mlp.fc1, h), { approximate: true }));
  return x.add(mlpOut);
}

function runVisionEncoder(
  { embeddings, encoder, postLayernorm }: VisionModel,
  pixelValues: np.Array, // [grid_t * grid_h * grid_w, 3, 14, 14]
  imageGridThw: ImageGridThw,
): np.Array {
  const [gridT, gridH, gridW] = imageGridThw;
  const numPatches = gridT * gridH * gridW;

  // Each flattened patch is already [3, 14, 14], so the conv reduces to a single linear projection.
  const patches = pixelValues.reshape([numPatches, 3 * VIS_PATCH_SIZE * VIS_PATCH_SIZE]); // [N, 588]
  const wFlat = embeddings.patchEmbedding.weight.ref.reshape([
    VIS_HIDDEN,
    3 * VIS_PATCH_SIZE * VIS_PATCH_SIZE,
  ]);
  let x = np.dot(patches, wFlat.transpose()).add(embeddings.patchEmbedding.bias!.ref); // [N, 1152]

  // Interpolate the learned 27x27 position embedding to the current patch grid.
  const posEmb = interpolateVisionPositionEmbedding(embeddings.positionEmbedding.weight.ref, gridH, gridW);
  if (gridT === 1) {
    x = x.add(posEmb);
  } else {
    x = x.add(np.repeat(posEmb.ref, gridT, 0));
  }

  const hPos = new Int32Array(numPatches);
  const wPos = new Int32Array(numPatches);
  for (let t = 0; t < gridT; t++) {
    for (let h = 0; h < gridH; h++) {
      for (let w = 0; w < gridW; w++) {
        const i = t * gridH * gridW + h * gridW + w;
        hPos[i] = h;
        wPos[i] = w;
      }
    }
  }
  const ropeEmb = buildVisionRope(
    np.array(hPos, { dtype: np.int32 }),
    np.array(wPos, { dtype: np.int32 }),
  );

  for (const layer of encoder.layers) {
    x = runVisionLayer(layer, x, ropeEmb);
  }
  return layerNorm(postLayernorm, x); // [N, 1152]
}

// ============================================================
// Projector: merge 2×2 spatial vision patches → fewer, wider tokens
// [729, 1152] → [169, 1024]
// ============================================================
function runProjector(
  { preNorm, linear1, linear2 }: Projector,
  visFeats: np.Array, // [729, 1152]
  imageGridThw: ImageGridThw,
): np.Array {
  // pre_norm weight is [1152] → normalize the vision features BEFORE spatial merge
  const normedVis = layerNorm(preNorm, visFeats); // [729, 1152]

  const [gridT, gridH, gridW] = imageGridThw;
  const llmGridH = Math.floor(gridH / PROJ_MERGE);
  const llmGridW = Math.floor(gridW / PROJ_MERGE);
  const grid = normedVis.reshape([gridT, gridH, gridW, VIS_HIDDEN]);

  // 2x2 spatial merge, matching the HF projector's rearrange.
  const merged = grid
    .reshape([gridT, llmGridH, PROJ_MERGE, llmGridW, PROJ_MERGE, VIS_HIDDEN])
    .transpose([0, 1, 3, 2, 4, 5])
    .reshape([gridT * llmGridH * llmGridW, PROJ_IN]);

  const h = nn.gelu(linear(linear1, merged), { approximate: true }); // [169, 4608]
  return linear(linear2, h); // [169, 1024]
}

// ============================================================
// LLM text decoder: GQA attention layer
//
// Missing ops:
//  - RMSNorm (called above)
//  - 3D RoPE (called here)
//  - GQA: jax-js dotProductAttention uses tile() (interleaved mapping).
//    The ERNIE4.5 model uses grouped mapping (queries 0-7 → KV head 0, etc.)
//    so we manually repeat KV heads with np.repeat() and call dotProductAttention
//    with equal head counts (no internal tiling).
// ============================================================
function runTextAttention(
  { qProj, kProj, vProj, oProj }: TextAttention,
  x: np.Array, // [T, text_hidden]
  tPos: np.Array, // [T] int32
  hPos: np.Array, // [T] int32
  wPos: np.Array, // [T] int32
  kvCache: KVCache | null,
  isCausal: boolean,
): [np.Array, KVCache] {
  const T = x.shape[0];

  const q = linear(qProj, x.ref).reshape([T, TEXT_Q_HEADS, TEXT_HEAD_DIM]); // [T,16,128]
  const kRaw = linear(kProj, x.ref).reshape([T, TEXT_KV_HEADS, TEXT_HEAD_DIM]); // [T,2,128]
  const vRaw = linear(vProj, x).reshape([T, TEXT_KV_HEADS, TEXT_HEAD_DIM]); // [T,2,128]

  // Apply 3D RoPE to Q and K
  const [qRot, kRot] = applyMRoPE(q, kRaw, tPos, hPos, wPos);

  // Extend KV cache (concatenate along sequence dimension)
  let fullK: np.Array;
  let fullV: np.Array;
  if (kvCache === null || kvCache.key.size === 0) {
    fullK = kRot;
    fullV = vRaw;
  } else {
    // Use .ref to keep old cache alive after concatenation
    fullK = np.concatenate([kvCache.key.ref, kRot], 0); // [S+T, 2, 128]
    fullV = np.concatenate([kvCache.value.ref, vRaw], 0); // [S+T, 2, 128]
    // Old cache will be replaced; caller must dispose it
  }
  const newCache: KVCache = { key: fullK, value: fullV };

  // GQA expansion: repeat each KV head TEXT_KV_GROUPS (=8) times along head axis.
  // np.repeat gives grouped pattern: [h0,h0,...,h0, h1,h1,...,h1] (correct for ERNIE4.5).
  // jax-js's internal tile gives interleaved pattern, so we expand manually.
  const kExp = np.repeat(fullK.ref, TEXT_KV_GROUPS, 1); // [S, 16, 128]
  const vExp = np.repeat(fullV.ref, TEXT_KV_GROUPS, 1); // [S, 16, 128]

  // Scaled dot-product attention with equal head counts (no internal GQA tiling)
  const attn = nn
    .dotProductAttention(qRot, kExp, vExp, { isCausal })
    .reshape([T, TEXT_Q_HEADS * TEXT_HEAD_DIM]); // [T, 2048]

  return [linear(oProj, attn), newCache]; // [T, 1024]
}

function runTextLayer(
  { inputLayernorm, selfAttn, postAttentionLayernorm, mlp }: TextLayer,
  x: np.Array, // [T, 1024]
  tPos: np.Array,
  hPos: np.Array,
  wPos: np.Array,
  kvCache: KVCache | null,
  isCausal: boolean,
): [np.Array, KVCache] {
  // Pre-norm → GQA → residual
  const [attnOut, newCache] = runTextAttention(
    selfAttn,
    rmsNorm(inputLayernorm, x.ref),
    tPos,
    hPos,
    wPos,
    kvCache,
    isCausal,
  );
  x = x.add(attnOut);

  // Pre-norm → SiLU gated MLP (gate_proj * up_proj → down_proj) → residual
  const h = rmsNorm(postAttentionLayernorm, x.ref);
  const gate = nn.silu(linear(mlp.gateProj, h.ref)); // [T, 3072]
  const up = linear(mlp.upProj, h); // [T, 3072]
  const mlpOut = linear(mlp.downProj, gate.mul(up)); // [T, 1024]

  return [x.add(mlpOut), newCache];
}

// ============================================================
// Prefill: process image + prompt in one causal forward pass
// Returns logits for the last position and the initial KV state.
// ============================================================
export function prefill(
  { visual, mlpAR, model, lmHead }: PaddleOCRModel,
  pixelValues: np.Array, // [grid_t * grid_h * grid_w, 3, 14, 14]
  imageGridThw: ImageGridThw,
  prefixTokenIds: number[], // tokens before vision tokens (im_start, user, \n, vis_start)
  suffixTokenIds: number[], // tokens after vision tokens (\n, OCR:, im_end, …, assistant, \n)
): { logits: np.Array; state: PaddleOCRState } {
  // Vision pipeline
  const visFeats = runVisionEncoder(visual.visionModel, pixelValues, imageGridThw);
  const imgEmbs = runProjector(mlpAR, visFeats, imageGridThw);

  // Text token embeddings for prefix and suffix
  const prefixIdArr = np.array(new Int32Array(prefixTokenIds), { dtype: np.int32 });
  const suffixIdArr = np.array(new Int32Array(suffixTokenIds), { dtype: np.int32 });
  const prefixEmb = model.embedTokens.weight.ref.slice(prefixIdArr.ref); // [nPrefix, 1024]
  const suffixEmb = model.embedTokens.weight.ref.slice(suffixIdArr.ref); // [nSuffix, 1024]
  prefixIdArr.dispose();
  suffixIdArr.dispose();

  // Full input: [prefix | image_features | suffix]
  const fullEmb = np.concatenate([prefixEmb, imgEmbs, suffixEmb], 0);
  const T = fullEmb.shape[0];

  // Build 3D position IDs for the full sequence
  // prefix: flat 1D (t=h=w=i)
  // image tokens: 3D multimodal positions over the merged image grid
  // suffix: flat 1D starting after the image's spatial extent
  const nP = prefixTokenIds.length;
  const [llmGridT, llmGridH, llmGridW] = [
    imageGridThw[0],
    imageGridThw[1] / PROJ_MERGE,
    imageGridThw[2] / PROJ_MERGE,
  ];
  const nI = llmGridT * llmGridH * llmGridW;
  const nS = suffixTokenIds.length;

  const tArr = new Int32Array(T);
  const hArr = new Int32Array(T);
  const wArr = new Int32Array(T);

  for (let i = 0; i < nP; i++) tArr[i] = hArr[i] = wArr[i] = i;

  for (let t = 0; t < llmGridT; t++) {
    for (let row = 0; row < llmGridH; row++) {
      for (let col = 0; col < llmGridW; col++) {
        const idx = nP + t * llmGridH * llmGridW + row * llmGridW + col;
        tArr[idx] = nP + t;
        hArr[idx] = nP + row;
        wArr[idx] = nP + col;
      }
    }
  }

  const afterImg = nP + Math.max(llmGridT, llmGridH, llmGridW);
  for (let i = 0; i < nS; i++) {
    tArr[nP + nI + i] = hArr[nP + nI + i] = wArr[nP + nI + i] = afterImg + i;
  }

  const tPos = np.array(tArr, { dtype: np.int32 });
  const hPos = np.array(hArr, { dtype: np.int32 });
  const wPos = np.array(wArr, { dtype: np.int32 });

  // Forward through 18 decoder layers (causal attention — no future tokens visible)
  let x = fullEmb;
  const kvCaches: KVCache[] = [];
  for (const layer of model.layers) {
    let kv: KVCache;
    [x, kv] = runTextLayer(layer, x, tPos.ref, hPos.ref, wPos.ref, null, true);
    kvCaches.push(kv);
  }
  tPos.dispose();
  hPos.dispose();
  wPos.dispose();

  // Logits for the last position (next token prediction)
  const lastX = rmsNorm(model.norm, x.slice([-1])); // [1, 1024]
  const logits = np.dot(lastX, lmHead.weight.ref.transpose()); // [1, vocab]
  console.log(`[prefill] seqLen=${T} prefixLen=${prefixTokenIds.length} suffixLen=${suffixTokenIds.length}`);

  return {
    logits,
    state: {
      kvCaches,
      decodeStep: 0,
      textPosBase: afterImg + nS, // 1D position of first generated token
    },
  };
}

// ============================================================
// Decode step: one new token using KV cache
// ============================================================
export function decodeStep(
  { model, lmHead }: PaddleOCRModel,
  tokenId: number,
  state: PaddleOCRState,
): { logits: np.Array; state: PaddleOCRState } {
  const pos = state.textPosBase + state.decodeStep;
  const posArr = np.array(new Int32Array([pos]), { dtype: np.int32 }); // [1]

  // Token embedding lookup
  const tokenIdArr = np.array(new Int32Array([tokenId]), { dtype: np.int32 });
  let x = model.embedTokens.weight.ref.slice(tokenIdArr.ref); // [1, 1024]
  tokenIdArr.dispose();

  // One forward pass per layer, extending KV cache
  const newCaches: KVCache[] = [];
  for (let i = 0; i < model.layers.length; i++) {
    let kv: KVCache;
    [x, kv] = runTextLayer(
      model.layers[i],
      x,
      posArr.ref,
      posArr.ref,
      posArr.ref,
      state.kvCaches[i],
      false, // not causal — we only have one query token, attending to full KV cache
    );
    // Old cache will no longer be referenced once newCaches replaces kvCaches
    newCaches.push(kv);
  }
  posArr.dispose();

  const normed = rmsNorm(model.norm, x); // [1, 1024]
  const logits = np.dot(normed, lmHead.weight.ref.transpose()); // [1, vocab]

  return {
    logits,
    state: { kvCaches: newCaches, decodeStep: state.decodeStep + 1, textPosBase: state.textPosBase },
  };
}

// ============================================================
// Tokenizer — Minimal BPE encode/decode for ERNIE-4.5
//
// The model uses a SentencePiece-compatible BPE where:
//   - Word-initial spaces → ▁ (U+2581) prefix (SentencePiece convention)
//   - Unknown bytes fall back to <0xNN> byte tokens
// ============================================================

export type Tokenizer = {
  idToToken: string[];
  tokenToId: Map<string, number>;
  mergeRanks: Map<string, number>;
  /** <|begin_of_sentence|> = 100273 — opens each conversation turn */
  beginSentenceId: number;
  /** <|end_of_sentence|> = 100272 — closes user turn (also used as generation stop) */
  endSentenceId: number;
  /** <0x0A> = 23 — byte token for \n in plain text */
  newlineId: number;
  /** </s> = 2 — EOS used to stop generation */
  eosId: number;
  /** <|TEXT_END|> = 101315 — structured text terminator used by OCR/table outputs */
  textEndId: number;
};

export async function loadTokenizer(
  url: string,
  onProgress?: (msg: string) => void,
): Promise<Tokenizer> {
  onProgress?.("Downloading tokenizer…");
  const data = await cachedFetch(url);
  const json = JSON.parse(new TextDecoder().decode(data));

  const vocabSize = Object.keys(json.model.vocab).length + (json.added_tokens?.length ?? 0);
  const idToToken: string[] = new Array(Math.max(vocabSize, 110000)).fill("");
  const tokenToId = new Map<string, number>();

  for (const [token, id] of Object.entries(json.model.vocab as Record<string, number>)) {
    idToToken[id] = token;
    tokenToId.set(token, id);
  }
  for (const t of (json.added_tokens ?? []) as Array<{ id: number; content: string }>) {
    idToToken[t.id] = t.content;
    tokenToId.set(t.content, t.id);
  }

  const mergeRanks = new Map<string, number>();
  for (let i = 0; i < (json.model.merges?.length ?? 0); i++) {
    mergeRanks.set(json.model.merges[i] as string, i);
  }

  // Locate special token IDs; try multiple spellings (tokenizer format may vary)
  const findId = (...keys: string[]): number => {
    for (const k of keys) {
      const id = tokenToId.get(k);
      if (id !== undefined) return id;
    }
    return 0;
  };

  return {
    idToToken,
    tokenToId,
    mergeRanks,
    beginSentenceId: findId("<|begin_of_sentence|>") || BEGIN_SENTENCE_ID,
    endSentenceId: findId("<|end_of_sentence|>") || END_SENTENCE_ID,
    newlineId: findId("<0x0A>") || NL_BYTE_ID,
    eosId: findId("</s>") || EOS_ID,
    textEndId: findId("<|TEXT_END|>"),
  };
}

/** Decode token IDs → UTF-8 text. Handles ▁ word boundaries and <0xNN> byte tokens.
 * <|LOC_N|> tokens (spotting bounding box coordinates) are passed through as-is. */
export function decodeTokens(ids: number[], tok: Tokenizer): string {
  const bytes: number[] = [];
  for (const id of ids) {
    const token = tok.idToToken[id];
    if (!token) continue;
    // Pass through location tokens for spotting task output
    if (token.startsWith("<|LOC_") && token.endsWith("|>")) {
      for (const b of new TextEncoder().encode(token)) bytes.push(b);
      continue;
    }
    // Skip other control tokens (begin/end_of_sentence, IMAGE_PLACEHOLDER, etc.)
    if (token.startsWith("<|") && token.endsWith("|>")) continue;

    const byteMatch = token.match(/^<0x([0-9A-Fa-f]{2})>$/);
    if (byteMatch) {
      bytes.push(parseInt(byteMatch[1], 16));
      continue;
    }

    // SentencePiece word boundary: ▁ (U+2581) → space prefix
    const text = token.startsWith("\u2581") ? " " + token.slice(1) : token;
    for (const b of new TextEncoder().encode(text)) bytes.push(b);
  }
  return new TextDecoder().decode(new Uint8Array(bytes)).trimStart();
}

/** BPE-encode a text string (SentencePiece ▁ convention). */
export function encodeText(text: string, tok: Tokenizer): number[] {
  if (!text) return [];
  const normalized = text.replaceAll(" ", "\u2581");
  const ids: number[] = [];

  // Initial units: each byte as its char or <0xNN>
  const units: string[] = Array.from(new TextEncoder().encode(normalized), (b) => {
    const ch = String.fromCharCode(b);
    return tok.tokenToId.has(ch) ? ch : `<0x${b.toString(16).padStart(2, "0").toUpperCase()}>`;
  });

  // BPE merge loop: repeatedly find the pair with lowest merge rank
  while (units.length > 1) {
    let best = Infinity;
    let bestI = -1;
    for (let i = 0; i < units.length - 1; i++) {
      const rank = tok.mergeRanks.get(units[i] + " " + units[i + 1]);
      if (rank !== undefined && rank < best) {
        best = rank;
        bestI = i;
      }
    }
    if (bestI === -1) break;
    units.splice(bestI, 1, units[bestI] + units[bestI + 1]);
    units.splice(bestI + 1, 1);
  }

  for (const u of units) {
    const id = tok.tokenToId.get(u);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

/** Build the OCR prompt token sequence (prefix = before image, suffix = after image).
 *
 * ERNIE-4.5 / PaddleOCR-VL-1.5 chat template (from chat_template.jinja):
 *
 *   <|begin_of_sentence|>User: <|IMAGE_START|>[169 × image feature vectors]<|IMAGE_END|>OCR:\nAssistant:\n
 *
 * Template source:
 *   cls_token + "User: " + IMAGE_START + IMAGE_PLACEHOLDER + IMAGE_END + task + "\n" + "Assistant:\n"
 *
 * The Hugging Face processor expands IMAGE_PLACEHOLDER to 169 repeated image tokens, then the model
 * replaces those token embeddings with projected image features. In this implementation we inject the
 * 169 image embeddings directly, but we still keep the surrounding IMAGE_START / IMAGE_END text tokens
 * and the trailing assistant newline because the model was trained with that layout.
 *
 * Token breakdown:
 *   prefix = [<|begin_of_sentence|>, ...BPE("User:"), <|IMAGE_START|>]
 *   suffix = [<|IMAGE_END|>, ...BPE("OCR:"), \n, ...BPE("Assistant:"), \n]
 */
export function buildOcrPrompt(
  tok: Tokenizer,
  task: "ocr" | "spotting" = "ocr",
): { prefixIds: number[]; suffixIds: number[] } {
  const userPrefixIds = encodeText("User:", tok);
  const assistantPrefixIds = encodeText("Assistant:", tok);

  // Task instruction (between image and assistant turn)
  const instrIds = task === "spotting" ? encodeText("Spotting:", tok) : encodeText("OCR:", tok);
  const imageStartId = tok.tokenToId.get("<|IMAGE_START|>") ?? tok.tokenToId.get("<|vision_start|>") ?? 0;
  const imageEndId = tok.tokenToId.get("<|IMAGE_END|>") ?? tok.tokenToId.get("<|vision_end|>") ?? 0;

  console.log(
    "[buildOcrPrompt] task=%s beginSentenceId=%d newlineId=%d eosId=%d imageStartId=%d imageEndId=%d " +
    "userPrefixIds=%s instrIds=%s assistantPrefixIds=%s",
    task, tok.beginSentenceId, tok.newlineId, tok.eosId,
    imageStartId, imageEndId,
    JSON.stringify(userPrefixIds), JSON.stringify(instrIds), JSON.stringify(assistantPrefixIds),
  );

  // prefix: <|begin_of_sentence|>User:<|IMAGE_START|>   (then 169 image feature vectors follow)
  const prefixIds = [tok.beginSentenceId, ...userPrefixIds, imageStartId];

  // suffix: <|IMAGE_END|>OCR:\nAssistant:\n   (model generates assistant body from here)
  const suffixIds = [imageEndId, ...instrIds, tok.newlineId, ...assistantPrefixIds, tok.newlineId];

  return { prefixIds, suffixIds };
}

// ============================================================
// Image preprocessing (browser canvas API)
// Resize to 384×384 and normalize with OPENAI CLIP mean/std.
// From image_processing_paddleocr_vl.py: image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
// Normalization: (pixel/255 - mean) / std   (NOT SigLIP [-1,1] style)
// ============================================================
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]; // R, G, B
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

function smartResize(
  height: number,
  width: number,
  factor = VIS_PATCH_SIZE * PROJ_MERGE,
  minPixels = 28 * 28 * 130,
  maxPixels = 28 * 28 * 1280,
): [number, number] {
  if (height < factor) {
    width = Math.round(width * factor / height);
    height = factor;
  }
  if (width < factor) {
    height = Math.round(height * factor / width);
    width = factor;
  }
  if (Math.max(height, width) / Math.min(height, width) > 200) {
    throw new Error(`Aspect ratio too extreme: ${width}x${height}`);
  }

  let hBar = Math.round(height / factor) * factor;
  let wBar = Math.round(width / factor) * factor;
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    hBar = Math.floor(height / beta / factor) * factor;
    wBar = Math.floor(width / beta / factor) * factor;
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    hBar = Math.ceil(height * beta / factor) * factor;
    wBar = Math.ceil(width * beta / factor) * factor;
  }
  return [hBar, wBar];
}

export function preprocessImage(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
): PreprocessedImage {
  const srcH =
    source instanceof HTMLImageElement ? source.naturalHeight :
    source instanceof HTMLCanvasElement ? source.height : source.height;
  const srcW =
    source instanceof HTMLImageElement ? source.naturalWidth :
    source instanceof HTMLCanvasElement ? source.width : source.width;
  const [resizedH, resizedW] = smartResize(srcH, srcW);

  const canvas = document.createElement("canvas");
  canvas.width = resizedW;
  canvas.height = resizedH;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(source as CanvasImageSource, 0, 0, resizedW, resizedH);
  const { data } = ctx.getImageData(0, 0, resizedW, resizedH);

  const gridT = 1;
  const gridH = resizedH / VIS_PATCH_SIZE;
  const gridW = resizedW / VIS_PATCH_SIZE;
  const numPatches = gridT * gridH * gridW;
  const patchArea = VIS_PATCH_SIZE * VIS_PATCH_SIZE;
  const patches = new Float32Array(numPatches * 3 * patchArea);

  for (let py = 0; py < gridH; py++) {
    for (let px = 0; px < gridW; px++) {
      const patchIndex = py * gridW + px;
      const patchBase = patchIndex * 3 * patchArea;
      for (let dy = 0; dy < VIS_PATCH_SIZE; dy++) {
        for (let dx = 0; dx < VIS_PATCH_SIZE; dx++) {
          const y = py * VIS_PATCH_SIZE + dy;
          const x = px * VIS_PATCH_SIZE + dx;
          const srcBase = (y * resizedW + x) * 4;
          const patchOffset = dy * VIS_PATCH_SIZE + dx;
          patches[patchBase + patchOffset] = (data[srcBase] / 255 - CLIP_MEAN[0]) / CLIP_STD[0];
          patches[patchBase + patchArea + patchOffset] = (data[srcBase + 1] / 255 - CLIP_MEAN[1]) / CLIP_STD[1];
          patches[patchBase + 2 * patchArea + patchOffset] = (data[srcBase + 2] / 255 - CLIP_MEAN[2]) / CLIP_STD[2];
        }
      }
    }
  }

  return {
    pixelValues: np.array(patches, {
      shape: [numPatches, 3, VIS_PATCH_SIZE, VIS_PATCH_SIZE],
      dtype: np.float32,
    }),
    imageGridThw: [gridT, gridH, gridW],
  };
}

// ============================================================
// Weight loading
// ============================================================

export const MODEL_URL =
  "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/model.safetensors";
export const TOKENIZER_URL =
  "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/tokenizer.json";

const weightMapper = new WeightMapper({
  prefix: {
    "visual.vision_model.": "visual.visionModel.",
    "mlp_AR.": "mlpAR.",
  },
  substring: {
    ".self_attn.q_proj": ".selfAttn.qProj",
    ".self_attn.k_proj": ".selfAttn.kProj",
    ".self_attn.v_proj": ".selfAttn.vProj",
    ".self_attn.o_proj": ".selfAttn.oProj",
    ".self_attn.out_proj": ".selfAttn.outProj",
    ".input_layernorm": ".inputLayernorm",
    ".post_attention_layernorm": ".postAttentionLayernorm",
    ".mlp.gate_proj": ".mlp.gateProj",
    ".mlp.up_proj": ".mlp.upProj",
    ".mlp.down_proj": ".mlp.downProj",
    ".layer_norm1": ".layerNorm1",
    ".layer_norm2": ".layerNorm2",
    ".patch_embedding": ".patchEmbedding",
    ".position_embedding": ".positionEmbedding",
    ".post_layernorm": ".postLayernorm",
    ".pre_norm": ".preNorm",
    ".linear_1": ".linear1",
    ".linear_2": ".linear2",
    "embed_tokens": "embedTokens",
    "lm_head": "lmHead",
  },
});

export function fromSafetensors(file: safetensors.File): PaddleOCRModel {
  const mapped = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};

  for (const [key, value] of Object.entries(mapped)) {
    // Skip the pooling head (not needed for generation) and flexible-size position embeddings
    if (key.includes(".head.") || key.includes("packing_position")) continue;

    if (value.dtype === "F32" || value.dtype === "BF16") {
      // BF16 tensors are already converted to Float32Array by the safetensors loader
      hydrated[key] = np.array(value.data as Float32Array, {
        dtype: np.float32,
        shape: value.shape,
      });
    } else if (value.dtype === "I64") {
      continue; // integer metadata (e.g., batch norm running stats)
    } else {
      console.warn(`PaddleOCR: skipping ${key} with unsupported dtype ${value.dtype}`);
    }
  }

  return safetensors.toNested(hydrated) as PaddleOCRModel;
}

// ============================================================
// High-level OCR inference
// ============================================================

export async function runOCR(
  model: PaddleOCRModel,
  tok: Tokenizer,
  image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  maxNewTokens = 512,
  onToken?: (partial: string) => void,
  task: "ocr" | "spotting" = "ocr",
  signal?: AbortSignal,
): Promise<string> {
  const { pixelValues, imageGridThw } = preprocessImage(image);
  const { prefixIds, suffixIds } = buildOcrPrompt(tok, task);

  // Prefill: process image + prompt together
  // Note: pixelValues is consumed (freed) inside prefill → runVisionEncoder → slice, so no dispose here.
  let { logits, state } = prefill(model, pixelValues, imageGridThw, prefixIds, suffixIds);

  const generatedIds: number[] = [];
  const stopIds = new Set([2, tok.eosId, tok.endSentenceId, tok.textEndId].filter((id) => id !== 0));

  // Greedy decode
  // NOTE: logits.data() internally calls this.dispose(), so never call dispose() after data().
  let logitsConsumed = false;
  for (let step = 0; step < maxNewTokens; step++) {
    if (signal?.aborted) {
      if (!logitsConsumed) logits.dispose();
      throw new DOMException("OCR generation stopped", "AbortError");
    }

    const data = (await logits.data()) as Float32Array; // data() disposes logits internally
    logitsConsumed = true;

    // argmax over vocabulary
    let nextId = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > bestVal) {
        bestVal = data[i];
        nextId = i;
      }
    }

    // Debug: log first 5 tokens to console
    if (step < 5) {
      // Top-5 tokens
      const top5 = Array.from(data)
        .map((v, i) => ({ v, i }))
        .sort((a, b) => b.v - a.v)
        .slice(0, 5)
        .map(({ v, i }) => `${i}(${tok.idToToken[i] ?? "?"}):${v.toFixed(2)}`);
      console.log(`[decode step=${step}] nextId=${nextId} token="${tok.idToToken[nextId] ?? "?"}" top5=[${top5}]`);
    }

    // Stop at any terminal control token used by PaddleOCR-VL outputs.
    if (stopIds.has(nextId)) break;
    generatedIds.push(nextId);

    onToken?.(decodeTokens(generatedIds, tok));

    ({ logits, state } = decodeStep(model, nextId, state));
    logitsConsumed = false;
  }
  // If we hit maxNewTokens without EOS, the final decodeStep logits was never data()'d
  if (!logitsConsumed) logits.dispose();

  return decodeTokens(generatedIds, tok);
}

/**
 * Standalone test script for PaddleOCR-VL-1.5 model loading & forward pass.
 *
 * Run with:  cd website && pnpm tsx src/routes/paddle-ocr/test-model.ts
 *
 * Caches the model to /tmp/paddle-ocr-model.safetensors so re-runs are fast.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

// ── Constants (copied from inference.ts) ─────────────────────────────────────
const VIS_HIDDEN = 1152;
const VIS_HEADS = 16;
const VIS_HEAD_DIM = VIS_HIDDEN / VIS_HEADS; // 72
const VIS_INTERMEDIATE = 4304;
const VIS_PATCH_SIZE = 14;
const TEXT_HIDDEN = 1024;
const TEXT_LAYERS = 18;
const TEXT_Q_HEADS = 16;
const TEXT_KV_HEADS = 2;
const TEXT_HEAD_DIM = 128;
const ROPE_THETA = 500000;
const MROPE_SECTION: [number, number, number] = [16, 24, 24];

const MODEL_URL =
  "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/model.safetensors";
const TOKENIZER_URL =
  "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/tokenizer.json";
const CACHE_DIR = "/tmp/paddle-ocr-cache";
const MODEL_CACHE = path.join(CACHE_DIR, "model.safetensors");
const TOK_CACHE = path.join(CACHE_DIR, "tokenizer.json");

// ── Helpers ───────────────────────────────────────────────────────────────────
function ok(label: string) {
  console.log(`  ✓ ${label}`);
}
function fail(label: string, detail: string) {
  console.error(`  ✗ ${label}: ${detail}`);
  process.exitCode = 1;
}
function section(title: string) {
  console.log(`\n── ${title} ${"─".repeat(50 - title.length)}`);
}
function approxEq(a: number, b: number, tol = 1e-3): boolean {
  return Math.abs(a - b) < tol;
}

async function downloadCached(url: string, cachePath: string): Promise<Buffer> {
  if (fs.existsSync(cachePath)) {
    console.log(`  Cached: ${cachePath}`);
    return fs.readFileSync(cachePath);
  }
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  console.log(`  Downloading ${url} …`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
  const buf = Buffer.from(await resp.arrayBuffer());
  fs.writeFileSync(cachePath, buf);
  console.log(`  Saved to ${cachePath}`);
  return buf;
}

// ── 1. BF16 Conversion ────────────────────────────────────────────────────────
section("1. BF16 → Float32 conversion");
{
  // bf16 bit pattern: sign=0, exp=127 (0x7F), mantissa=0 → value=1.0
  // BF16 hex for 1.0: 0x3F80
  const bf16_1 = new Uint16Array([0x3f80]); // 1.0
  const bf16_2 = new Uint16Array([0x4000]); // 2.0  (0x40000000 >> 16)
  const bf16_neg = new Uint16Array([0xbf80]); // -1.0

  const out = new Float32Array(3);
  const outI32 = new Int32Array(out.buffer);
  const src = new Uint16Array([bf16_1[0], bf16_2[0], bf16_neg[0]]);
  for (let i = 0; i < 3; i++) outI32[i] = src[i] << 16;

  if (approxEq(out[0], 1.0)) ok("bf16 1.0 → 1.0");
  else fail("bf16 1.0", `got ${out[0]}`);
  if (approxEq(out[1], 2.0)) ok("bf16 2.0 → 2.0");
  else fail("bf16 2.0", `got ${out[1]}`);
  if (approxEq(out[2], -1.0)) ok("bf16 -1.0 → -1.0");
  else fail("bf16 -1.0", `got ${out[2]}`);
}

// ── 2. Weight mapper ──────────────────────────────────────────────────────────
section("2. Weight mapper key translations");
{
  const wm = new WeightMapper({
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

  const cases: [string, string][] = [
    [
      "visual.vision_model.encoder.layers.0.self_attn.q_proj.weight",
      "visual.visionModel.encoder.layers.0.selfAttn.qProj.weight",
    ],
    [
      "visual.vision_model.encoder.layers.0.self_attn.out_proj.weight",
      "visual.visionModel.encoder.layers.0.selfAttn.outProj.weight",
    ],
    [
      "model.layers.0.self_attn.o_proj.weight",
      "model.layers.0.selfAttn.oProj.weight",
    ],
    ["model.embed_tokens.weight", "model.embedTokens.weight"],
    ["lm_head.weight", "lmHead.weight"],
    ["mlp_AR.pre_norm.weight", "mlpAR.preNorm.weight"],
    ["mlp_AR.linear_1.weight", "mlpAR.linear1.weight"],
  ];

  for (const [input, expected] of cases) {
    const got = wm.mapKey(input);
    if (got === expected) ok(`${input.slice(0, 40)}`);
    else fail(`map(${input.slice(0, 40)})`, `expected "${expected}", got "${got}"`);
  }
}

// ── 3. Load safetensors header & check keys ───────────────────────────────────
section("3. Safetensors — key names & shapes");
{
  let modelBuf: Buffer;
  try {
    modelBuf = await downloadCached(MODEL_URL, MODEL_CACHE);
  } catch (e) {
    console.log(`  SKIP (download failed: ${e})`);
    process.exit(process.exitCode ?? 0);
  }

  const file = safetensors.parse(new Uint8Array(modelBuf.buffer, modelBuf.byteOffset, modelBuf.byteLength));
  const keys = Object.keys(file.tensors);
  console.log(`  Total tensors: ${keys.length}`);
  console.log(`  First 10 raw keys:`);
  keys.slice(0, 10).forEach((k) => console.log(`    ${k}  ${JSON.stringify(file.tensors[k].shape)}`));

  // Check dtype distribution
  const dtypeCounts: Record<string, number> = {};
  for (const t of Object.values(file.tensors)) {
    dtypeCounts[t.dtype] = (dtypeCounts[t.dtype] ?? 0) + 1;
  }
  console.log(`  DType counts: ${JSON.stringify(dtypeCounts)}`);

  // Required keys (raw, before mapping)
  const required = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "visual.vision_model.embeddings.patch_embedding.weight",
    "visual.vision_model.embeddings.position_embedding.weight",
    "mlp_AR.pre_norm.weight",
    "mlp_AR.linear_1.weight",
    "mlp_AR.linear_2.weight",
  ];
  for (const k of required) {
    if (file.tensors[k]) {
      ok(`${k}  ${JSON.stringify(file.tensors[k].shape)}`);
    } else {
      fail(`missing key`, k);
      // Try to find a close match
      const close = keys.filter((x) => x.includes(k.split(".")[0])).slice(0, 3);
      if (close.length) console.log(`    Close matches: ${close.join(", ")}`);
    }
  }

  // Check shape of lm_head vs embed_tokens (tied weights?)
  const lmh = file.tensors["lm_head.weight"];
  const emb = file.tensors["model.embed_tokens.weight"];
  if (lmh && emb) {
    const tied = JSON.stringify(lmh.shape) === JSON.stringify(emb.shape);
    console.log(`  lm_head shape: ${lmh.shape}, embed_tokens shape: ${emb.shape} (${tied ? "same shape" : "DIFFERENT"})`);
    if (lmh.data instanceof Float32Array && emb.data instanceof Float32Array) {
      const same = lmh.data[0] === emb.data[0] && lmh.data[1] === emb.data[1];
      console.log(`  lm_head[0:2]=${[lmh.data[0], lmh.data[1]]}, embed[0:2]=${[emb.data[0], emb.data[1]]} (${same ? "TIED — same data!" : "different"})`);
    }
  }

  // Check expected attention dimensions
  const qw = file.tensors["model.layers.0.self_attn.q_proj.weight"];
  const kw = file.tensors["model.layers.0.self_attn.k_proj.weight"];
  if (qw) {
    const expectedQ = [TEXT_Q_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN]; // [2048, 1024]
    if (JSON.stringify(qw.shape) === JSON.stringify(expectedQ)) ok(`q_proj shape ${qw.shape}`);
    else fail(`q_proj shape`, `expected ${expectedQ}, got ${qw.shape}`);
  }
  if (kw) {
    const expectedK = [TEXT_KV_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN]; // [256, 1024]
    if (JSON.stringify(kw.shape) === JSON.stringify(expectedK)) ok(`k_proj shape ${kw.shape}`);
    else fail(`k_proj shape`, `expected ${expectedK}, got ${kw.shape}`);
  }
}

// ── 4. jax-js numeric ops ─────────────────────────────────────────────────────
section("4. jax-js RMSNorm & LayerNorm unit tests");
{
  const devices = await init("wasm");
  if (!devices.includes("wasm")) {
    console.log("  SKIP (wasm not available)");
  } else {
    defaultDevice("wasm");

    // RMSNorm: x = [1, 2, 3, 4], weight = [1, 1, 1, 1]
    // rms = sqrt(mean([1,4,9,16])) = sqrt(7.5) = 2.7386
    // normalized = [0.365, 0.730, 1.095, 1.460]
    {
      const x = np.array([1, 2, 3, 4], { dtype: np.float32 });
      const w = np.array([1, 1, 1, 1], { dtype: np.float32 });
      const variance = np.mean(np.square(x.ref), -1, { keepdims: true });
      const normed = x.div(np.sqrt(variance.add(1e-6))).mul(w);
      const out = normed.js() as number[];
      const expected = [1 / Math.sqrt(7.5), 2 / Math.sqrt(7.5), 3 / Math.sqrt(7.5), 4 / Math.sqrt(7.5)];
      const close = out.every((v, i) => approxEq(v, expected[i], 1e-4));
      if (close) ok(`RMSNorm([1,2,3,4]) = [${out.map((v) => v.toFixed(4))}]`);
      else fail("RMSNorm", `got ${out}, expected ${expected.map((v) => v.toFixed(4))}`);
    }

    // MRoPE: check that cos/sin shapes are [T, 1, 128]
    {
      const T = 3;
      const halfDim = TEXT_HEAD_DIM / 2; // 64
      const [sT, sH, sW] = MROPE_SECTION;

      const makeInvFreq = (size: number, dimOffset: number): np.Array => {
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = 1.0 / Math.pow(ROPE_THETA, (2 * (i + dimOffset)) / TEXT_HEAD_DIM);
        }
        return np.array(data);
      };
      const freqT = makeInvFreq(sT, 0);
      const freqH = makeInvFreq(sH, sT);
      const freqW = makeInvFreq(sW, sT + sH);

      const tPos = np.array(new Int32Array([0, 1, 2]), { dtype: np.int32 });
      const hPos = np.array(new Int32Array([0, 1, 2]), { dtype: np.int32 });
      const wPos = np.array(new Int32Array([0, 1, 2]), { dtype: np.int32 });

      const angT = tPos.astype(np.float32).reshape([-1, 1]).mul(freqT.reshape([1, -1]));
      const angH = hPos.astype(np.float32).reshape([-1, 1]).mul(freqH.reshape([1, -1]));
      const angW = wPos.astype(np.float32).reshape([-1, 1]).mul(freqW.reshape([1, -1]));

      const angles = np.concatenate([angT, angH, angW], -1); // [T, 64]
      const cosHalf = np.cos(angles.ref);
      const sinHalf = np.sin(angles);
      const cosA = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, TEXT_HEAD_DIM]);
      const sinA = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, TEXT_HEAD_DIM]);

      const cosShape = cosA.shape;
      const sinShape = sinA.shape;
      if (JSON.stringify(cosShape) === JSON.stringify([T, 1, TEXT_HEAD_DIM]))
        ok(`cosA shape = ${cosShape}`);
      else fail("cosA shape", `got ${cosShape}, expected [${T},1,${TEXT_HEAD_DIM}]`);

      // Check position 0: all angles = 0, so cos = 1, sin = 0
      const cosData = cosA.js() as number[][][];
      const allOne = cosData[0][0].every((v) => approxEq(v, 1.0, 1e-5));
      if (allOne) ok("cos(0) = 1 for all dims");
      else fail("cos(0)", `got ${cosData[0][0].slice(0, 5)}`);

      sinA.dispose();
    }

    // Test that embedding lookup [vocab, hidden].slice([1]) → [1, hidden]
    {
      const vocab = 10, hidden = 4;
      const w = np.array(
        Float32Array.from({ length: vocab * hidden }, (_, i) => i),
        { shape: [vocab, hidden], dtype: np.float32 },
      );
      const idx = np.array(new Int32Array([3]), { dtype: np.int32 });
      const emb = w.slice(idx);
      const out = emb.js() as number[][];
      const expected = [12, 13, 14, 15]; // row 3: indices 12..15
      const ok2 = JSON.stringify(out) === JSON.stringify([expected]);
      if (ok2) ok(`embedding lookup slice([3]) = ${JSON.stringify(out)}`);
      else fail("embedding lookup", `got ${JSON.stringify(out)}, expected ${JSON.stringify([expected])}`);
    }
  }
}

// ── 5. Tokenizer spot checks ──────────────────────────────────────────────────
section("5. Tokenizer loading & special token IDs");
{
  let tokBuf: Buffer;
  try {
    tokBuf = await downloadCached(TOKENIZER_URL, TOK_CACHE);
  } catch (e) {
    console.log(`  SKIP (download failed: ${e})`);
    process.exit(process.exitCode ?? 0);
  }

  const json = JSON.parse(tokBuf.toString("utf8"));
  const allTokens: Record<string, number> = { ...(json.model?.vocab ?? {}) };
  for (const t of (json.added_tokens ?? []) as Array<{ id: number; content: string }>) {
    allTokens[t.content] = t.id;
  }

  const VISION_START_TOKEN_ID = 101305;
  const VISION_END_TOKEN_ID = 101306;
  const IMAGE_TOKEN_ID = 100295;

  for (const [name, expectedId, searchStr] of [
    ["<|im_start|>", null, "<|im_start|>"],
    ["<|im_end|>", null, "<|im_end|>"],
    ["<vision_start>", VISION_START_TOKEN_ID, "<vision_start>"],
    ["<vision_end>", VISION_END_TOKEN_ID, "<vision_end>"],
    ["<image>", IMAGE_TOKEN_ID, "<image>"],
  ] as [string, number | null, string][]) {
    const id = allTokens[searchStr];
    if (id === undefined) {
      fail(`token "${name}"`, "NOT FOUND in vocab");
      // Search for similar tokens
      const similar = Object.keys(allTokens).filter((k) => k.toLowerCase().includes(name.toLowerCase().replace(/[<>|]/g, ""))).slice(0, 3);
      if (similar.length) console.log(`    Similar tokens: ${similar.map((k) => `"${k}":${allTokens[k]}`).join(", ")}`);
    } else if (expectedId !== null && id !== expectedId) {
      fail(`token "${name}"`, `expected ID ${expectedId}, got ${id}`);
    } else {
      ok(`"${name}" = ${id}`);
    }
  }

  // Check all tokens needed by loadTokenizer
  for (const [name, expectedId] of [
    ["<|TEXT_START|>", 101314],
    ["<|TEXT_END|>", 101315],
    ["<nl>", 101313],
    ["</s>", 2],
    ["<s>", 1],
  ] as [string, number][]) {
    const id = allTokens[name];
    if (id === expectedId) ok(`"${name}" = ${id}`);
    else if (id !== undefined) fail(`"${name}"`, `expected ${expectedId}, got ${id}`);
    else fail(`"${name}"`, "NOT FOUND");
  }

  // Check "user" and "assistant" tokens
  for (const role of ["user", "assistant"]) {
    const plain = allTokens[role];
    const sp = allTokens["\u2581" + role]; // ▁user
    console.log(`  "${role}": plain=${plain ?? "missing"}, ▁${role}=${sp ?? "missing"}`);
  }
}

// ── 6. Image normalization check ──────────────────────────────────────────────
section("6. Image normalization");
{
  // SigLIP preprocessor_config.json: image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5]
  // Formula: (pixel/255 - mean) / std = (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
  const pixel0 = 0;   // black → expected -1.0
  const pixel255 = 255; // white → expected 1.0
  const pixel128 = 128; // ~gray → expected 0.00392

  const norm = (p: number) => p / 127.5 - 1.0;
  if (approxEq(norm(pixel0), -1.0)) ok("black pixel → -1.0");
  else fail("black pixel", `got ${norm(pixel0)}`);
  if (approxEq(norm(pixel255), 1.0, 0.01)) ok("white pixel → 1.0");
  else fail("white pixel", `got ${norm(pixel255)}`);
  console.log(`  gray pixel (128) → ${norm(pixel128).toFixed(4)} (expected ~0.004)`);
}

console.log("\n── Done ────────────────────────────────────────────────────────────");

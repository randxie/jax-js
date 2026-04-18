import { nn, numpy as np } from "../dist/index.js";
import * as st from "../packages/loaders/src/safetensors.ts";

type Linear = {
  weight: np.Array;
};

type RMSNorm = {
  weight: np.Array;
};

type QwenLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  selfAttn: {
    qNorm: RMSNorm;
    kNorm: RMSNorm;
    qProj: Linear;
    kProj: Linear;
    vProj: Linear;
    oProj: Linear;
  };
  mlp: {
    gateProj: Linear;
    upProj: Linear;
    downProj: Linear;
  };
};

export type FunASRNanoQwen = {
  embedTokens: np.Array;
  embedTokensData: Float16Array<ArrayBuffer>;
  norm: RMSNorm;
  layers: QwenLayer[];
  hiddenSize: number;
  numHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  ropeTheta: number;
  rmsNormEps: number;
  eosTokenIds: Set<number>;
};

type KVCache = {
  key: np.Array;
  value: np.Array;
};

type QwenDecodeState = {
  caches: KVCache[];
  seqLen: number;
};

function tensorToArray(tensor: st.Tensor): np.Array {
  if (tensor.dtype !== "F16") {
    throw new Error(`Expected F16 safetensor, got ${tensor.dtype}`);
  }
  return np.array(
    Float32Array.from(tensor.data as Float16Array<ArrayBuffer>),
    {
    dtype: np.float32,
    shape: tensor.shape,
    },
  );
}

function runLinear({ weight }: Linear, x: np.Array): np.Array {
  return np.dot(x.ref, weight.ref.transpose());
}

function runRMSNorm(
  { weight }: RMSNorm,
  x: np.Array,
  eps: number,
): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const var_ = np.var_(x.ref, -1, { correction: 0, keepdims: true });
  x = x.mul(weight.ref).div(np.sqrt(var_.add(eps)));
  return x.astype(dtype);
}

function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.neg(), x1], -1);
}

function applyRope(
  q: np.Array,
  k: np.Array,
  offset: number,
  theta: number,
): [np.Array, np.Array] {
  const [tq, , d] = q.shape;
  const [tk] = k.shape;
  if (tq !== tk) throw new Error("RoPE requires matching sequence lengths");
  const halfD = d / 2;
  const ds = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(ds.mul((-Math.log(theta) * 2) / d));
  const ts = np.arange(tq).add(offset).astype(np.float32).reshape([tq, 1]);
  const freqs = ts.mul(invFreq);
  const emb = np.concatenate([freqs.ref, freqs], -1).astype(q.dtype);
  const cos = np.expandDims(np.cos(emb.ref), 1);
  const sin = np.expandDims(np.sin(emb), 1);
  return [
    q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref)),
    k.ref.mul(cos).add(rotateHalf(k).mul(sin)),
  ];
}

function repeatKV(x: np.Array, repeats: number): np.Array {
  const [t, h, d] = x.shape;
  return np
    .tile(x.reshape([t, h, 1, d]), [1, 1, repeats, 1])
    .reshape([t, h * repeats, d]);
}

function runAttention(model: FunASRNanoQwen, layer: QwenLayer, x: np.Array): np.Array {
  const [t] = x.shape;
  let q = runLinear(layer.selfAttn.qProj, x).reshape([
    t,
    model.numHeads,
    model.headDim,
  ]).astype(np.float32);
  let k = runLinear(layer.selfAttn.kProj, x).reshape([
    t,
    model.numKeyValueHeads,
    model.headDim,
  ]).astype(np.float32);
  let v = runLinear(layer.selfAttn.vProj, x).reshape([
    t,
    model.numKeyValueHeads,
    model.headDim,
  ]).astype(np.float32);

  q = runRMSNorm(layer.selfAttn.qNorm, q, model.rmsNormEps);
  k = runRMSNorm(layer.selfAttn.kNorm, k, model.rmsNormEps);
  [q, k] = applyRope(q, k, 0, model.ropeTheta);
  k = repeatKV(k, model.numHeads / model.numKeyValueHeads);
  v = repeatKV(v, model.numHeads / model.numKeyValueHeads);

  const attn = nn.dotProductAttention(q, k.ref, v.ref, { isCausal: true });
  return runLinear(
    layer.selfAttn.oProj,
    attn.reshape([t, model.numHeads * model.headDim]),
  );
}

function runAttentionStep(
  model: FunASRNanoQwen,
  layer: QwenLayer,
  x: np.Array,
  cache: KVCache,
  offset: number,
): [np.Array, KVCache] {
  const [t] = x.shape;
  let q = runLinear(layer.selfAttn.qProj, x).reshape([
    t,
    model.numHeads,
    model.headDim,
  ]).astype(np.float32);
  let k = runLinear(layer.selfAttn.kProj, x).reshape([
    t,
    model.numKeyValueHeads,
    model.headDim,
  ]).astype(np.float32);
  let v = runLinear(layer.selfAttn.vProj, x).reshape([
    t,
    model.numKeyValueHeads,
    model.headDim,
  ]).astype(np.float32);

  q = runRMSNorm(layer.selfAttn.qNorm, q, model.rmsNormEps);
  k = runRMSNorm(layer.selfAttn.kNorm, k, model.rmsNormEps);
  [q, k] = applyRope(q, k, offset, model.ropeTheta);

  const fullK = cache.key.size > 0 ? np.concatenate([cache.key.ref, k], 0) : k;
  const fullV = cache.value.size > 0 ? np.concatenate([cache.value.ref, v], 0) : v;
  const repeatedK = repeatKV(fullK, model.numHeads / model.numKeyValueHeads);
  const repeatedV = repeatKV(fullV, model.numHeads / model.numKeyValueHeads);

  const totalLen = fullK.shape[0];
  const maskDelta = np
    .arange(totalLen)
    .sub(np.arange(t).reshape([t, 1]))
    .sub(offset);
  const mask = maskDelta.lessEqual(0);
  const attn = nn.dotProductAttention(q, repeatedK, repeatedV, { mask });

  return [
    runLinear(
      layer.selfAttn.oProj,
      attn.reshape([t, model.numHeads * model.headDim]),
    ),
    { key: fullK, value: fullV },
  ];
}

function runMLP(layer: QwenLayer, x: np.Array): np.Array {
  const gate = nn.silu(runLinear(layer.mlp.gateProj, x).astype(np.float32));
  const up = runLinear(layer.mlp.upProj, x).astype(np.float32);
  return runLinear(layer.mlp.downProj, gate.mul(up));
}

export function runFunASRNanoQwen(
  model: FunASRNanoQwen,
  inputEmbeds: np.Array,
): np.Array {
  let hidden = inputEmbeds.astype(np.float32);
  for (const layer of model.layers) {
    const residual1 = hidden.ref;
    hidden = runRMSNorm(layer.inputLayernorm, hidden, model.rmsNormEps);
    hidden = residual1.add(runAttention(model, layer, hidden));

    const residual2 = hidden.ref;
    hidden = runRMSNorm(layer.postAttentionLayernorm, hidden, model.rmsNormEps);
    hidden = residual2.add(runMLP(layer, hidden));
  }
  return runRMSNorm(model.norm, hidden, model.rmsNormEps);
}

function prefillFunASRNanoQwen(
  model: FunASRNanoQwen,
  inputEmbeds: np.Array,
): { hidden: np.Array; state: QwenDecodeState } {
  let hidden = inputEmbeds.astype(np.float32);
  const caches: KVCache[] = [];
  for (const layer of model.layers) {
    const residual1 = hidden.ref;
    hidden = runRMSNorm(layer.inputLayernorm, hidden, model.rmsNormEps);
    hidden = residual1.add(runAttention(model, layer, hidden));

    const residual2 = hidden.ref;
    hidden = runRMSNorm(layer.postAttentionLayernorm, hidden, model.rmsNormEps);
    hidden = residual2.add(runMLP(layer, hidden));

    const normed = runRMSNorm(layer.inputLayernorm, residual1, model.rmsNormEps);
    let k = runLinear(layer.selfAttn.kProj, normed).reshape([
      inputEmbeds.shape[0],
      model.numKeyValueHeads,
      model.headDim,
    ]).astype(np.float32);
    let v = runLinear(layer.selfAttn.vProj, normed).reshape([
      inputEmbeds.shape[0],
      model.numKeyValueHeads,
      model.headDim,
    ]).astype(np.float32);
    k = runRMSNorm(layer.selfAttn.kNorm, k, model.rmsNormEps);
    [, k] = applyRope(
      np.zeros([inputEmbeds.shape[0], model.numHeads, model.headDim], {
        dtype: np.float32,
      }),
      k,
      0,
      model.ropeTheta,
    );
    caches.push({ key: k, value: v });
  }
  return {
    hidden: runRMSNorm(model.norm, hidden, model.rmsNormEps),
    state: { caches, seqLen: inputEmbeds.shape[0] },
  };
}

function decodeFunASRNanoQwenStep(
  model: FunASRNanoQwen,
  tokenEmbeds: np.Array,
  state: QwenDecodeState,
): { hidden: np.Array; state: QwenDecodeState } {
  let hidden = tokenEmbeds.astype(np.float32);
  const caches: KVCache[] = [];
  for (let i = 0; i < model.layers.length; i++) {
    const layer = model.layers[i];
    const residual1 = hidden.ref;
    hidden = runRMSNorm(layer.inputLayernorm, hidden, model.rmsNormEps);
    const [attn, cache] = runAttentionStep(
      model,
      layer,
      hidden,
      state.caches[i],
      state.seqLen,
    );
    hidden = residual1.add(attn);

    const residual2 = hidden.ref;
    hidden = runRMSNorm(layer.postAttentionLayernorm, hidden, model.rmsNormEps);
    hidden = residual2.add(runMLP(layer, hidden));
    caches.push(cache);
  }
  return {
    hidden: runRMSNorm(model.norm, hidden, model.rmsNormEps),
    state: { caches, seqLen: state.seqLen + tokenEmbeds.shape[0] },
  };
}

function gatherEmbeddingRows(
  data: Float16Array<ArrayBuffer>,
  hiddenSize: number,
  ids: number[],
): Float16Array<ArrayBuffer> {
  const out = new Float16Array(ids.length * hiddenSize);
  for (let i = 0; i < ids.length; i++) {
    const start = ids[i] * hiddenSize;
    out.set(data.subarray(start, start + hiddenSize), i * hiddenSize);
  }
  return out;
}

export function embedTokenIds(model: FunASRNanoQwen, ids: number[]): np.Array {
  return np.array(gatherEmbeddingRows(model.embedTokensData, model.hiddenSize, ids), {
    dtype: np.float16,
    shape: [ids.length, model.hiddenSize],
  });
}

export async function buildFunASRNanoInputEmbeds(
  model: FunASRNanoQwen,
  sourceIds: number[],
  fbankBeg: number,
  fakeTokenLen: number,
  audioEmbeds: np.Array,
): Promise<np.Array> {
  const data = gatherEmbeddingRows(model.embedTokensData, model.hiddenSize, sourceIds);
  const audioData = await audioEmbeds.data();
  for (let i = 0; i < fakeTokenLen; i++) {
    const dst = (fbankBeg + i) * model.hiddenSize;
    const src = i * model.hiddenSize;
    for (let j = 0; j < model.hiddenSize; j++) {
      data[dst + j] = audioData[src + j];
    }
  }
  return np.array(data, {
    dtype: np.float16,
    shape: [sourceIds.length, model.hiddenSize],
  });
}

export async function generateFunASRNanoGreedy(
  model: FunASRNanoQwen,
  baseEmbeds: np.Array,
  maxNewTokens: number,
): Promise<number[]> {
  const generated: number[] = [];
  let { hidden, state } = prefillFunASRNanoQwen(model, baseEmbeds);
  for (let step = 0; step < maxNewTokens; step++) {
    const lastHidden = hidden.slice([-1]).astype(np.float32);
    const logits = np.dot(lastHidden, model.embedTokens.ref.transpose()).reshape([
      model.embedTokens.shape[0],
    ]);
    const token = Number(np.argmax(logits).js());
    generated.push(token);
    if (model.eosTokenIds.has(token)) break;
    const nextEmbeds = embedTokenIds(model, [token]);
    ({ hidden, state } = decodeFunASRNanoQwenStep(model, nextEmbeds, state));
  }
  return generated;
}

export function decodeGeneratedText(
  tokenizer: { decode(tokens: number[]): string },
  tokens: number[],
): string {
  return tokenizer
    .decode(tokens)
    .replaceAll("<|im_end|>", "")
    .replaceAll("<|endoftext|>", "")
    .trim();
}

export function loadFunASRNanoQwenFromBuffers(
  modelData: Uint8Array<ArrayBuffer> | ArrayBuffer,
  configText: string,
  generationConfigText: string,
): FunASRNanoQwen {
  const file = st.parse(modelData);
  const nested = st.toNested(
    Object.fromEntries(
      Object.entries(file.tensors).map(([key, value]) => [key, tensorToArray(value)]),
    ),
  );
  const embedTensor = file.tensors["model.embed_tokens.weight"];
  if (!embedTensor || embedTensor.dtype !== "F16") {
    throw new Error("Missing model.embed_tokens.weight");
  }

  const config = JSON.parse(configText);
  const generationConfig = JSON.parse(generationConfigText);

  return {
    embedTokens: nested.model.embed_tokens.weight,
    embedTokensData: embedTensor.data as Float16Array<ArrayBuffer>,
    norm: { weight: nested.model.norm.weight },
    layers: nested.model.layers.map((layer: any) => ({
      inputLayernorm: { weight: layer.input_layernorm.weight },
      postAttentionLayernorm: { weight: layer.post_attention_layernorm.weight },
      selfAttn: {
        qNorm: { weight: layer.self_attn.q_norm.weight },
        kNorm: { weight: layer.self_attn.k_norm.weight },
        qProj: { weight: layer.self_attn.q_proj.weight },
        kProj: { weight: layer.self_attn.k_proj.weight },
        vProj: { weight: layer.self_attn.v_proj.weight },
        oProj: { weight: layer.self_attn.o_proj.weight },
      },
      mlp: {
        gateProj: { weight: layer.mlp.gate_proj.weight },
        upProj: { weight: layer.mlp.up_proj.weight },
        downProj: { weight: layer.mlp.down_proj.weight },
      },
    })),
    hiddenSize: config.hidden_size,
    numHeads: config.num_attention_heads,
    numKeyValueHeads: config.num_key_value_heads,
    headDim: config.head_dim,
    ropeTheta: config.rope_theta,
    rmsNormEps: config.rms_norm_eps,
    eosTokenIds: new Set(
      Array.isArray(generationConfig.eos_token_id)
        ? generationConfig.eos_token_id
        : [generationConfig.eos_token_id],
    ),
  };
}

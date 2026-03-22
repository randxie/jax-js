import * as fs from "node:fs";

import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

import {
  buildOcrPrompt,
  decodeStep,
  decodeTokens,
  fromSafetensors,
  prefill,
  type Tokenizer,
} from "./inference";

const CACHE_DIR = "/tmp/paddle-ocr-cache";
const MODEL_CACHE = `${CACHE_DIR}/model.safetensors`;
const TOK_CACHE = `${CACHE_DIR}/tokenizer.json`;
const IMG_META = `${CACHE_DIR}/test-snippet.meta.json`;
const IMG_BIN = `${CACHE_DIR}/test-snippet.f32`;

function loadLocalTokenizer(path: string): Tokenizer {
  const json = JSON.parse(fs.readFileSync(path, "utf8"));
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
    beginSentenceId: findId("<|begin_of_sentence|>"),
    endSentenceId: findId("<|end_of_sentence|>"),
    newlineId: findId("<0x0A>"),
    eosId: findId("</s>"),
    textEndId: findId("<|TEXT_END|>"),
  };
}

async function main() {
  const devices = await init("wasm");
  if (!devices.includes("wasm")) {
    throw new Error("wasm backend unavailable");
  }
  defaultDevice("wasm");

  const tok = loadLocalTokenizer(TOK_CACHE);
  const modelBuf = fs.readFileSync(MODEL_CACHE);
  const file = safetensors.parse(new Uint8Array(modelBuf.buffer, modelBuf.byteOffset, modelBuf.byteLength) as any);
  const model = fromSafetensors(file);

  const meta = JSON.parse(fs.readFileSync(IMG_META, "utf8")) as {
    shape: [number, number, number, number];
    imageGridThw: [number, number, number];
  };
  const imgRaw = fs.readFileSync(IMG_BIN);
  const img = np.array(
    new Float32Array(imgRaw.buffer, imgRaw.byteOffset, imgRaw.byteLength / 4) as any,
    { shape: meta.shape, dtype: np.float32 },
  );

  const { prefixIds, suffixIds } = buildOcrPrompt(tok, "ocr");
  let { logits, state } = prefill(model, img, meta.imageGridThw, prefixIds, suffixIds);

  const generatedIds: number[] = [];
  for (let step = 0; step < 80; step++) {
    const data = (await logits.data()) as Float32Array;
    let nextId = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > bestVal) {
        bestVal = data[i];
        nextId = i;
      }
    }
    const top5 = Array.from(data)
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 5)
      .map(({ v, i }) => `${i}(${tok.idToToken[i] ?? "?"}):${v.toFixed(2)}`);
    console.log(`[step ${step}] next=${nextId} token=${JSON.stringify(tok.idToToken[nextId] ?? "?")} top5=${top5.join(", ")}`);

    if (nextId === tok.eosId || nextId === tok.endSentenceId) break;
    generatedIds.push(nextId);
    ({ logits, state } = decodeStep(model, nextId, state));
  }

  console.log("\nOCR:\n");
  console.log(decodeTokens(generatedIds, tok));
}

void main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

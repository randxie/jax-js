import { readFileSync } from "node:fs";
import path from "node:path";

import { defaultDevice, numpy as np } from "../dist/index.js";
import { ONNXModel } from "../packages/onnx/dist/index.js";
import { HuggingFaceBPE } from "../packages/loaders/src/tokenizers.ts";

import { extractFunASRNanoSpeech } from "./funasr_nano_frontend.ts";
import { prepareFunASRNanoSourceIds } from "./funasr_nano_prepare.ts";
import {
  buildFunASRNanoInputEmbeds,
  decodeGeneratedText,
  generateFunASRNanoGreedy,
  loadFunASRNanoQwenFromBuffers,
} from "./funasr_nano_qwen.ts";

function getFlag(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index >= 0 && index + 1 < process.argv.length) {
    return process.argv[index + 1];
  }
  return fallback;
}

async function main() {
  defaultDevice("cpu");

  const samplePath = path.resolve(
    getFlag(
      "--sample",
      ".download/Fun-ASR-Nano-2512/funasr_nano_encoder_sample.json",
    )!,
  );
  const onnxPath = path.resolve(
    getFlag(
      "--encoder",
      ".download/Fun-ASR-Nano-2512/funasr_nano_encoder.onnx",
    )!,
  );
  const llmPath = path.resolve(
    getFlag(
      "--llm",
      ".artifacts/local/funasr_nano_llm_fp16.safetensors",
    )!,
  );
  const qwenDir = path.resolve(
    getFlag("--qwen-dir", ".download/Fun-ASR-Nano-2512/Qwen3-0.6B")!,
  );
  const audioPath = getFlag(
    "--audio",
    ".download/Fun-ASR-Nano-2512/example/zh.mp3",
  )!;
  const maxNewTokens = Number(getFlag("--max-new-tokens", "16"));

  const sample = JSON.parse(readFileSync(samplePath, "utf8"));
  const waveform = new Float32Array(sample.waveform);
  const { speech, shape } = await extractFunASRNanoSpeech(waveform, sample.frontend);
  const speechLengths = np.array(new Int32Array([shape[1]]), {
    dtype: np.int32,
    shape: [1],
  });

  const encoder = new ONNXModel(new Uint8Array(readFileSync(onnxPath)));
  const encoderOut = encoder.run({
    speech: np.array(speech, { shape }),
    speech_lengths: speechLengths,
  }).encoder_out;

  const tokenizer = HuggingFaceBPE.fromBinary(
    new Uint8Array(readFileSync(path.join(qwenDir, "tokenizer.json"))),
  );
  const prepared = prepareFunASRNanoSourceIds(tokenizer, audioPath, shape[1]);

  const qwen = loadFunASRNanoQwen(
    new Uint8Array(readFileSync(llmPath)),
    readFileSync(path.join(qwenDir, "config.json"), "utf8"),
    readFileSync(path.join(qwenDir, "generation_config.json"), "utf8"),
  );
  const baseEmbeds = await buildFunASRNanoInputEmbeds(
    qwen,
    prepared.sourceIds,
    prepared.fbankBeg,
    prepared.fakeTokenLen,
    encoderOut,
  );

  const generatedIds = await generateFunASRNanoGreedy(
    qwen,
    baseEmbeds,
    maxNewTokens,
  );
  const transcript = decodeGeneratedText(tokenizer, generatedIds);

  console.log(
    JSON.stringify(
      {
        sample: samplePath,
        encoder: onnxPath,
        llm: llmPath,
        speech_shape: shape,
        fbank_beg: prepared.fbankBeg,
        fake_token_len: prepared.fakeTokenLen,
        generated_ids: generatedIds,
        transcript,
      },
      null,
      2,
    ),
  );
}

await main();

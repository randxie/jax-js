import { readFileSync } from "node:fs";
import path from "node:path";

import { numpy as np } from "../dist/index.js";
import { ONNXModel } from "../packages/onnx/dist/index.js";

function getFlag(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length
    ? process.argv[index + 1]
    : fallback;
}

async function main() {
  const modelPath = path.resolve(
    getFlag(
      "--model",
      ".download/Fun-ASR-Nano-2512/funasr_nano_encoder.onnx",
    ),
  );
  const samplePath = path.resolve(
    getFlag(
      "--sample",
      ".download/Fun-ASR-Nano-2512/funasr_nano_encoder_sample.json",
    ),
  );

  const modelBytes = readFileSync(modelPath);
  const sample = JSON.parse(readFileSync(samplePath, "utf8"));

  const speech = np.array(new Float32Array(sample.speech), {
    shape: sample.speech_shape,
  });
  const speechLengths = np.array(new Int32Array(sample.speech_lengths), {
    dtype: np.int32,
    shape: [sample.speech_lengths.length],
  });

  const model = new ONNXModel(
    new Uint8Array(
      modelBytes.buffer,
      modelBytes.byteOffset,
      modelBytes.byteLength,
    ),
  );
  const out = model.run({ speech, speech_lengths: speechLengths });

  const got = await out.encoder_out.data();
  const gotLens = Array.from(await out.encoder_out_lens.data());
  const expected = new Float32Array(sample.encoder_out);

  let maxAbs = 0;
  let meanAbs = 0;
  for (let i = 0; i < got.length; i++) {
    const delta = Math.abs(got[i] - expected[i]);
    if (delta > maxAbs) maxAbs = delta;
    meanAbs += delta;
  }
  meanAbs /= got.length;

  console.log(
    JSON.stringify({
      shape: out.encoder_out.shape,
      lens: gotLens,
      maxAbs,
      meanAbs,
    }),
  );
}

await main();

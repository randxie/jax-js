import { readFileSync } from "node:fs";
import path from "node:path";

import { extractFunASRNanoSpeech } from "./funasr_nano_frontend.ts";

function getSamplePath(): string {
  const index = process.argv.indexOf("--sample");
  if (index >= 0 && index + 1 < process.argv.length) {
    return process.argv[index + 1];
  }
  if (process.argv.length >= 3 && !process.argv[2].startsWith("--")) {
    return process.argv[2];
  }
  return ".download/Fun-ASR-Nano-2512/funasr_nano_encoder_sample.json";
}

async function main() {
  const samplePath = path.resolve(getSamplePath());
  const sample = JSON.parse(readFileSync(samplePath, "utf8"));

  const waveform = new Float32Array(sample.waveform);
  const expected = new Float32Array(sample.speech);
  const { speech, shape } = await extractFunASRNanoSpeech(
    waveform,
    sample.frontend,
  );

  let maxAbs = 0;
  let meanAbs = 0;
  for (let i = 0; i < speech.length; i++) {
    const delta = Math.abs(speech[i] - expected[i]);
    if (delta > maxAbs) maxAbs = delta;
    meanAbs += delta;
  }
  meanAbs /= speech.length;

  console.log(JSON.stringify({ shape, maxAbs, meanAbs }));
}

await main();

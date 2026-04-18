import { readFileSync } from "node:fs";
import path from "node:path";

import { tokenizers } from "../packages/loaders/src/index.ts";

import { prepareFunASRNanoSourceIds } from "./funasr_nano_prepare.ts";

function getFlag(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index >= 0 && index + 1 < process.argv.length) {
    return process.argv[index + 1];
  }
  return fallback;
}

function expectEqual(name: string, actual: number[], expected: number[]) {
  const sameLength = actual.length === expected.length;
  const sameValues =
    sameLength && actual.every((value, i) => value === expected[i]);
  if (!sameValues) {
    throw new Error(
      `${name} mismatch\nexpected=${JSON.stringify(expected)}\nactual=${JSON.stringify(actual)}`,
    );
  }
}

async function main() {
  const tokenizerPath = path.resolve(
    getFlag(
      "--tokenizer",
      ".download/Fun-ASR-Nano-2512/Qwen3-0.6B/tokenizer.json",
    )!,
  );
  const speechLength = Number(getFlag("--speech-length", "94"));
  const audioPath = getFlag(
    "--audio",
    ".download/Fun-ASR-Nano-2512/example/zh.mp3",
  )!;

  const tokenizer = tokenizers.HuggingFaceBPE.fromBinary(
    new Uint8Array(readFileSync(tokenizerPath)),
  );

  const prepared = prepareFunASRNanoSourceIds(
    tokenizer,
    audioPath,
    speechLength,
  );
  const expectedSourceIds = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644,
    872, 198, 105761, 46670, 61443, 5122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    151645, 198, 151644, 77091, 198,
  ];
  expectEqual("source ids", prepared.sourceIds, expectedSourceIds);
  if (prepared.fbankBeg !== 18) {
    throw new Error(`fbankBeg mismatch: expected 18, got ${prepared.fbankBeg}`);
  }
  if (prepared.fakeTokenLen !== 12) {
    throw new Error(
      `fakeTokenLen mismatch: expected 12, got ${prepared.fakeTokenLen}`,
    );
  }

  console.log(
    JSON.stringify(
      {
        tokenizer: tokenizerPath,
        audio: audioPath,
        speech_length: speechLength,
        fbank_beg: prepared.fbankBeg,
        fake_token_len: prepared.fakeTokenLen,
        source_ids: prepared.sourceIds,
      },
      null,
      2,
    ),
  );
}

await main();

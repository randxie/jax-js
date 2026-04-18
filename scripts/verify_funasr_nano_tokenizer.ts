import { readFileSync } from "node:fs";
import path from "node:path";

import { tokenizers } from "../packages/loaders/src/index.ts";

import { buildFunASRNanoChatMLPrompt } from "./funasr_nano_prompt.ts";

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
      `${name} token mismatch\nexpected=${JSON.stringify(expected)}\nactual=${JSON.stringify(actual)}`,
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
  const audioPath = getFlag(
    "--audio",
    ".download/Fun-ASR-Nano-2512/example/zh.mp3",
  )!;

  const tokenizer = tokenizers.HuggingFaceBPE.fromBinary(
    new Uint8Array(readFileSync(tokenizerPath)),
  );

  const simplePrompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
  const simpleIds = tokenizer.encodeWithSpecialTokens(simplePrompt);
  const expectedSimple = [151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198];
  expectEqual("simple prompt", simpleIds, expectedSimple);

  const funasrPrompt = buildFunASRNanoChatMLPrompt(audioPath);
  const funasrIds = tokenizer.encodeWithSpecialTokens(funasrPrompt);
  const expectedFunASR = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644,
    872, 198, 105761, 46670, 61443, 90737, 91, 2468, 1055, 88225, 91, 29,
    15365, 12885, 12318, 359, 12, 1911, 49, 11250, 5652, 12, 17, 20, 16, 17,
    65182, 14, 23815, 16870, 18, 27, 91, 408, 1055, 88225, 91, 29, 151645,
    198, 151644, 77091, 198,
  ];
  expectEqual("funasr prompt", funasrIds, expectedFunASR);

  console.log(
    JSON.stringify(
      {
        tokenizer: tokenizerPath,
        audio: audioPath,
        simple_prompt_ids: simpleIds,
        funasr_prompt_ids: funasrIds,
        funasr_prompt_len: funasrIds.length,
      },
      null,
      2,
    ),
  );
}

await main();

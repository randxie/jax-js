import { buildFunASRNanoChatMLPrompt, type FunASRNanoPromptOptions } from "./funasr_nano_prompt.ts";

type TokenizerLike = {
  encodeWithSpecialTokens(text: string): number[];
};

const speechTagPattern = /(<\|startofspeech\|>.*?<\|endofspeech\|>)/g;

export function computeFunASRNanoFakeTokenLength(speechLength: number): number {
  let length = 1 + Math.floor((speechLength - 3 + 2) / 2);
  length = 1 + Math.floor((length - 3 + 2) / 2);
  return Math.floor((length - 1) / 2) + 1;
}

export function prepareFunASRNanoSourceIds(
  tokenizer: TokenizerLike,
  audioPath: string,
  speechLength: number,
  opts: FunASRNanoPromptOptions = {},
): {
  prompt: string;
  sourceIds: number[];
  fbankBeg: number;
  fakeTokenLen: number;
} {
  const prompt = buildFunASRNanoChatMLPrompt(audioPath, opts);
  const parts = prompt.split(speechTagPattern);
  const fakeTokenLen = computeFunASRNanoFakeTokenLength(speechLength);

  const sourceIds: number[] = [];
  let fbankBeg = -1;

  for (const part of parts) {
    if (!part) continue;
    if (!part.startsWith("<|startofspeech|>")) {
      sourceIds.push(...tokenizer.encodeWithSpecialTokens(part));
      continue;
    }
    fbankBeg = sourceIds.length;
    sourceIds.push(...new Array(fakeTokenLen).fill(0));
  }

  return { prompt, sourceIds, fbankBeg, fakeTokenLen };
}

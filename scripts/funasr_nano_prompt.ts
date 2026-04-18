export type FunASRNanoPromptOptions = {
  hotwords?: string[];
  language?: string | null;
  itn?: boolean;
  systemPrompt?: string;
  doThink?: boolean;
};

export function getFunASRNanoPrompt({
  hotwords = [],
  language = null,
  itn = true,
}: FunASRNanoPromptOptions = {}): string {
  let prompt = "";
  if (hotwords.length > 0) {
    prompt =
      "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n";
    prompt += `热词列表：[${hotwords.join(", ")}]\n`;
  }
  prompt += language === null ? "语音转写" : `语音转写成${language}`;
  if (!itn) prompt += "，不进行文本规整";
  return `${prompt}：`;
}

export function buildFunASRNanoChatMLPrompt(
  audioPath: string,
  {
    hotwords = [],
    language = null,
    itn = true,
    systemPrompt = "You are a helpful assistant.",
    doThink = true,
  }: FunASRNanoPromptOptions = {},
): string {
  const prompt = getFunASRNanoPrompt({ hotwords, language, itn });
  const userPrompt = `${prompt}<|startofspeech|>!${audioPath}<|endofspeech|>`;

  let sourceInput =
    `<|im_start|>system\n${systemPrompt}<|im_end|>\n` +
    `<|im_start|>user\n${userPrompt}<|im_end|>\n` +
    `<|im_start|>assistant\n`;

  if (!doThink) {
    sourceInput += "<think>\n\n</think>\n\n";
  }

  return sourceInput;
}

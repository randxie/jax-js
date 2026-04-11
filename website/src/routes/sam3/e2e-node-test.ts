/**
 * End-to-end verifier for the SAM-3 integration in commit 44642d8.
 *
 * This checks whether the local filesystem weights are compatible with the
 * current jax-js SAM-3 integration, and runs the parts that are possible in
 * Node.js without a browser WebGPU runtime.
 *
 * Default weights directory:
 *   /home/randxie/weights/sam3
 *
 * Run with:
 *   cd website && pnpm tsx src/routes/sam3/e2e-node-test.ts
 *   cd website && pnpm tsx src/routes/sam3/e2e-node-test.ts /path/to/weights
 */

import * as fs from "node:fs";
import * as path from "node:path";

const DEFAULT_WEIGHTS_DIR = "/home/randxie/weights/sam3";
const SAM3_CONTEXT_LENGTH = 32;

type TokenizerJson = {
  model?: {
    vocab?: Record<string, number>;
  };
  added_tokens?: Array<{ id: number; content: string }>;
};

function section(title: string) {
  console.log(`\n== ${title} ${"=".repeat(Math.max(1, 72 - title.length))}`);
}

function ok(message: string) {
  console.log(`  OK  ${message}`);
}

function fail(message: string): never {
  throw new Error(message);
}

function exists(filePath: string): boolean {
  return fs.existsSync(filePath);
}

function readTokenizer(tokenizerPath: string) {
  const json = JSON.parse(fs.readFileSync(tokenizerPath, "utf8")) as TokenizerJson;
  const vocab = json.model?.vocab ?? {};
  const tokenToId = new Map<string, number>();
  for (const [token, id] of Object.entries(vocab)) {
    tokenToId.set(token, id);
  }
  for (const item of json.added_tokens ?? []) {
    tokenToId.set(item.content, item.id);
  }
  return { tokenToId, vocabSize: Object.keys(vocab).length };
}

function tokenizePromptFromLocalTokenizer(
  prompt: string,
  tokenToId: Map<string, number>,
): Int32Array {
  const SOT = tokenToId.get("<|startoftext|>") ?? 49406;
  const EOT = tokenToId.get("<|endoftext|>") ?? 49407;
  const unk = tokenToId.get("<|endoftext|>") ?? 49407;

  const words = prompt
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  const tokens = new Int32Array(SAM3_CONTEXT_LENGTH);
  tokens[0] = SOT;
  const limit = SAM3_CONTEXT_LENGTH - 2;
  let cursor = 1;
  for (const word of words) {
    if (cursor > limit) break;
    tokens[cursor++] = tokenToId.get(word) ?? unk;
  }
  tokens[cursor] = EOT;
  return tokens;
}

function formatSize(filePath: string): string {
  const bytes = fs.statSync(filePath).size;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function main() {
  const weightsDir = path.resolve(process.argv[2] ?? DEFAULT_WEIGHTS_DIR);

  section("Environment");
  console.log(`  Repo integration commit: 44642d8 (sam3 init)`);
  console.log(`  Weights directory: ${weightsDir}`);

  if (!exists(weightsDir)) {
    fail(`Weights directory not found: ${weightsDir}`);
  }

  const imageEncoderPath = path.join(weightsDir, "sam3_image_encoder.onnx");
  const languageEncoderPath = path.join(weightsDir, "sam3_language_encoder.onnx");
  const decoderPath = path.join(weightsDir, "sam3_decoder.onnx");
  const tokenizerPath = path.join(weightsDir, "tokenizer.json");
  const safetensorsPath = path.join(weightsDir, "model.safetensors");
  const ptPath = path.join(weightsDir, "sam3.pt");

  section("Local files");
  for (const filePath of [tokenizerPath, safetensorsPath, ptPath]) {
    if (exists(filePath)) {
      ok(`${path.basename(filePath)} present (${formatSize(filePath)})`);
    }
  }
  for (const filePath of [imageEncoderPath, languageEncoderPath, decoderPath]) {
    if (exists(filePath)) {
      ok(`${path.basename(filePath)} present (${formatSize(filePath)})`);
    } else {
      console.log(`  MISS ${path.basename(filePath)}`);
    }
  }

  section("Tokenizer");
  if (exists(tokenizerPath)) {
    const { tokenToId, vocabSize } = readTokenizer(tokenizerPath);
    ok(`Loaded tokenizer.json with ${vocabSize} vocab entries`);
    const prompt = "a cat";
    const tokens = tokenizePromptFromLocalTokenizer(prompt, tokenToId);
    ok(`Local prompt tokenization for ${JSON.stringify(prompt)} produced ${tokens.length} ids`);
    console.log(`  Tokens: ${Array.from(tokens.slice(0, 6)).join(", ")} ...`);
  } else {
    console.log("  SKIP tokenizer.json not found");
  }

  section("Integration verdict");
  const haveAllOnnx =
    exists(imageEncoderPath) && exists(languageEncoderPath) && exists(decoderPath);

  if (!haveAllOnnx) {
    const foundHfCheckpoint = exists(safetensorsPath) || exists(ptPath);
    if (foundHfCheckpoint) {
      fail(
        [
          "SAM-3 integration is not runnable end-to-end from this local weights directory.",
          "Reason: commit 44642d8 loads three ONNX graphs, but this directory contains the Hugging Face checkpoint format instead.",
          "Present: tokenizer.json, model.safetensors, sam3.pt.",
          "Missing: sam3_image_encoder.onnx, sam3_language_encoder.onnx, sam3_decoder.onnx.",
        ].join(" "),
      );
    }
    fail(
      "SAM-3 integration is not runnable end-to-end because the required ONNX files are missing.",
    );
  }

  const [{ defaultDevice, init }, { ONNXModel }] = await Promise.all([
    import("@jax-js/jax"),
    import("@jax-js/onnx"),
  ]);

  section("jax-js backend");
  const devices = await init("cpu");
  defaultDevice(devices[0]);
  ok(`Initialized ${devices[0]} backend`);

  const decoderBytes = fs.readFileSync(decoderPath);
  const imageEncoderBytes = fs.readFileSync(imageEncoderPath);
  const languageEncoderBytes = fs.readFileSync(languageEncoderPath);

  const decoderModel = new ONNXModel(decoderBytes);
  const imageEncoderModel = new ONNXModel(imageEncoderBytes);
  const languageEncoderModel = new ONNXModel(languageEncoderBytes);

  ok(`Loaded decoder (${formatSize(decoderPath)})`);
  ok(`Loaded image encoder (${formatSize(imageEncoderPath)})`);
  ok(`Loaded language encoder (${formatSize(languageEncoderPath)})`);

  const decoderInputs = decoderModel.model.graph?.input?.map((x) => x.name) ?? [];
  const decoderOutputs = decoderModel.model.graph?.output?.map((x) => x.name) ?? [];
  const imageOutputs = imageEncoderModel.model.graph?.output?.map((x) => x.name) ?? [];
  const languageOutputs =
    languageEncoderModel.model.graph?.output?.map((x) => x.name) ?? [];

  ok(`Decoder graph inputs: ${decoderInputs.length}`);
  ok(`Decoder graph outputs: ${decoderOutputs.join(", ")}`);
  ok(`Image encoder graph outputs: ${imageOutputs.join(", ")}`);
  ok(`Language encoder graph outputs: ${languageOutputs.join(", ")}`);

  console.log("\nPASS SAM-3 ONNX assets are present and loadable. Full inference still requires the ONNX export path used by the browser demo.");
}

void main().catch((error) => {
  console.error(`\nFAIL ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});

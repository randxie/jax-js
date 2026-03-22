/**
 * Inspect the tokenizer JSON structure to find where special tokens are stored.
 * Run: cd website && node --import tsx/esm src/routes/paddle-ocr/inspect-tokenizer.ts
 */
import * as fs from "node:fs";
const CACHE_DIR = "/tmp/paddle-ocr-cache";
const TOK_CACHE = `${CACHE_DIR}/tokenizer.json`;
const TOK_CONFIG_CACHE = `${CACHE_DIR}/tokenizer_config.json`;

// Download tokenizer_config.json if not cached
if (!fs.existsSync(TOK_CONFIG_CACHE)) {
  console.log("Downloading tokenizer_config.json...");
  const resp = await fetch("https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/tokenizer_config.json");
  fs.writeFileSync(TOK_CONFIG_CACHE, Buffer.from(await resp.arrayBuffer()));
}
const tokConfig = JSON.parse(fs.readFileSync(TOK_CONFIG_CACHE, "utf8"));
console.log("tokenizer_config keys:", Object.keys(tokConfig));
console.log("bos_token:", tokConfig.bos_token);
console.log("eos_token:", tokConfig.eos_token);
console.log("pad_token:", tokConfig.pad_token);
console.log("unk_token:", tokConfig.unk_token);
console.log("image_token:", tokConfig.image_token);
console.log("extra_special_tokens:", JSON.stringify(tokConfig.extra_special_tokens, null, 2));
if (tokConfig.chat_template) {
  console.log("\nchat_template:\n", tokConfig.chat_template);
} else {
  console.log("No chat_template in tokenizer_config");
}

// Fetch processor_config.json for image preprocessing details
const PROC_CACHE = `${CACHE_DIR}/processor_config.json`;
if (!fs.existsSync(PROC_CACHE)) {
  console.log("Downloading processor_config.json...");
  const r = await fetch("https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/processor_config.json");
  if (r.ok) fs.writeFileSync(PROC_CACHE, Buffer.from(await r.arrayBuffer()));
  else console.log("  Not found:", r.status);
}
if (fs.existsSync(PROC_CACHE)) {
  const pc = JSON.parse(fs.readFileSync(PROC_CACHE, "utf8"));
  console.log("\nprocessor_config:", JSON.stringify(pc, null, 2));
}

// Fetch generation_config.json to find eos_token_id
const GEN_CACHE = `${CACHE_DIR}/generation_config.json`;
if (!fs.existsSync(GEN_CACHE)) {
  const r = await fetch("https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/resolve/main/generation_config.json");
  if (r.ok) fs.writeFileSync(GEN_CACHE, Buffer.from(await r.arrayBuffer()));
}
if (fs.existsSync(GEN_CACHE)) {
  console.log("\ngeneration_config:", fs.readFileSync(GEN_CACHE, "utf8"));
}

console.log("---");

if (!fs.existsSync(TOK_CACHE)) {
  console.error("Run test-model.ts first to download the tokenizer.");
  process.exit(1);
}

const json = JSON.parse(fs.readFileSync(TOK_CACHE, "utf8"));
const addedTokens: Array<{ id: number; content: string; special: boolean }> = json.added_tokens ?? [];

console.log(`Total added_tokens: ${addedTokens.length}`);
console.log(`Vocab size (model.vocab): ${Object.keys(json.model?.vocab ?? {}).length}`);

// Sort by ID
const sorted = [...addedTokens].sort((a, b) => a.id - b.id);

// Print high-ID tokens (> 99000) — this is where vision/special tokens live
console.log("\nAdded tokens with ID > 99000:");
sorted.filter((t) => t.id > 99000).forEach((t) => console.log(`  ${t.id}: ${JSON.stringify(t.content)}`));

// Search for patterns
const patterns = ["im_start", "im_end", "vision", "image", "newline", "bos", "eos", "pad", "sep", "cls"];
console.log("\nAdded tokens matching patterns:", patterns);
for (const t of sorted) {
  const low = t.content.toLowerCase();
  if (patterns.some((p) => low.includes(p))) {
    console.log(`  ${t.id}: ${JSON.stringify(t.content)}  special=${t.special}`);
  }
}

// Show the last 20 added tokens
console.log("\nLast 20 added tokens (highest IDs):");
sorted.slice(-20).forEach((t) => console.log(`  ${t.id}: ${JSON.stringify(t.content)}`));

// Post processor template (shows what tokens wrap input/output)
if (json.post_processor?.single) {
  console.log("\npost_processor.single:", JSON.stringify(json.post_processor.single));
}
if (json.post_processor?.pair) {
  console.log("post_processor.pair:", JSON.stringify(json.post_processor.pair));
}

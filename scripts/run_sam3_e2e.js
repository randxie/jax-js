#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import path from "node:path";

const python = process.env.SAM3_PYTHON || "/home/randxie/.venv/bin/python";
const weights = process.env.SAM3_WEIGHTS || "/home/randxie/weights/sam3";
const prompt = process.env.SAM3_PROMPT || "a cat";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const helper = path.join(repoRoot, "scripts", "sam3_e2e.py");

const proc = spawnSync(python, [helper, "--weights", weights, "--prompt", prompt], {
  cwd: repoRoot,
  encoding: "utf8",
  maxBuffer: 10 * 1024 * 1024,
});

if (proc.status !== 0) {
  process.stderr.write(proc.stdout || "");
  process.stderr.write(proc.stderr || "");
  console.error(`FAIL Python helper exited with status ${proc.status ?? "unknown"}`);
  process.exit(proc.status ?? 1);
}

const lines = (proc.stdout || "").trim().split("\n");
const marker = lines.find((line) => line.startsWith("RESULT_JSON="));
if (!marker) {
  process.stderr.write(proc.stdout || "");
  process.stderr.write(proc.stderr || "");
  console.error("FAIL Missing RESULT_JSON output from Python helper");
  process.exit(1);
}

const result = JSON.parse(marker.slice("RESULT_JSON=".length));

const checks = [
  [result.ok === true, "helper completed successfully"],
  [JSON.stringify(result.masks_shape) === JSON.stringify([1, 200, 288, 288]), "mask shape is [1,200,288,288]"],
  [JSON.stringify(result.boxes_shape) === JSON.stringify([1, 200, 4]), "box shape is [1,200,4]"],
  [JSON.stringify(result.scores_shape) === JSON.stringify([1, 200]), "score shape is [1,200]"],
  [result.masks_finite === true, "mask values are finite"],
  [result.boxes_finite === true, "box values are finite"],
  [result.scores_finite === true, "score values are finite"],
  [result.mask_abs_max > 0, "mask output is non-trivial"],
  [result.score_max > result.score_min, "scores have variation"],
  [result.box_min > -0.25 && result.box_max < 1.25, "boxes are in a reasonable normalized range"],
];

const failed = checks.filter(([ok]) => !ok);
if (failed.length > 0) {
  console.error("FAIL SAM-3 e2e verification failed:");
  for (const [, message] of failed) {
    console.error(`  - ${message}`);
  }
  console.error(JSON.stringify(result, null, 2));
  process.exit(1);
}

console.log("PASS SAM-3 local HF checkpoint ran end-to-end");
console.log(JSON.stringify(result, null, 2));

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

type VerifyMetrics = {
  shape?: number[];
  lens?: number[];
  maxAbs: number;
  meanAbs: number;
};

function getFlag(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length
    ? process.argv[index + 1]
    : undefined;
}

function requireFlag(name: string): string {
  const value = getFlag(name);
  if (!value) {
    throw new Error(`Missing required flag: ${name}`);
  }
  return value;
}

function runCommand(command: string, args: string[]): string {
  const result = spawnSync(command, args, {
    cwd: path.resolve("."),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout);
    process.stderr.write(result.stderr);
    throw new Error(
      `${command} ${args.join(" ")} failed with exit code ${result.status ?? "unknown"}`,
    );
  }
  return result.stdout.trim();
}

function parseLastJson(stdout: string): any {
  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      return JSON.parse(lines[i]);
    } catch {
      continue;
    }
  }
  throw new Error(`Expected JSON output, got:\n${stdout}`);
}

async function main() {
  const audioPath = path.resolve(requireFlag("--audio"));
  const expectedText = getFlag("--expected-text");
  const workDir = mkdtempSync(path.join(tmpdir(), "funasr-nano-"));
  const samplePath = path.join(workDir, "encoder_sample.json");

  try {
    runCommand("npx", ["pnpm", "build"]);

    const transcribeArgs = [
      "scripts/verify_funasr_nano_transcription.py",
      "--audio",
      audioPath,
    ];
    if (expectedText) {
      transcribeArgs.push("--expected-text", expectedText);
    }
    const transcript = parseLastJson(
      runCommand(".venv/bin/python", transcribeArgs),
    );

    runCommand(".venv/bin/python", [
      "scripts/export_funasr_nano_encoder.py",
      "--audio",
      audioPath,
      "--sample-out",
      samplePath,
    ]);

    const frontend = parseLastJson(
      runCommand("npx", [
        "pnpm",
        "exec",
        "tsx",
        "scripts/verify_funasr_nano_frontend.ts",
        "--sample",
        samplePath,
      ]),
    ) as VerifyMetrics;
    const encoder = parseLastJson(
      runCommand("npx", [
        "pnpm",
        "exec",
        "tsx",
        "scripts/run_funasr_nano_encoder.ts",
        "--sample",
        samplePath,
      ]),
    ) as VerifyMetrics;

    console.log(
      JSON.stringify(
        {
          audio: audioPath,
          transcript: transcript.transcript,
          normalized_transcript: transcript.normalized_transcript,
          expected_text: transcript.expected_text,
          normalized_expected_text: transcript.normalized_expected_text,
          matches: transcript.matches,
          frontend,
          encoder,
        },
        null,
        2,
      ),
    );
  } finally {
    rmSync(workDir, { force: true, recursive: true });
  }
}

await main();

import path from "node:path";

import { chromium } from "playwright";

function getFlag(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index >= 0 && index + 1 < process.argv.length) {
    return process.argv[index + 1];
  }
  return fallback;
}

function hasFlag(name: string): boolean {
  return process.argv.includes(name);
}

async function readPreText(page: any, title: string): Promise<string> {
  const block = page.locator("div").filter({ has: page.getByText(title, { exact: true }) });
  return (await block.locator("pre").textContent())?.trim() || "";
}

async function main() {
  const pageUrl = getFlag("--url", "http://127.0.0.1:4175/funasr-nano")!;
  const audioPath = path.resolve(
    getFlag("--audio", ".artifacts/local/funasr_nano_zh.wav")!,
  );
  const recordMs = Number(getFlag("--record-ms", "2000"));
  const timeoutMs = Number(getFlag("--timeout-ms", "180000"));
  const headless = !hasFlag("--headed");

  const browser = await chromium.launch({
    headless,
    args: [
      "--enable-unsafe-webgpu",
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      `--use-file-for-fake-audio-capture=${audioPath}`,
      "--autoplay-policy=no-user-gesture-required",
      "--no-sandbox",
    ],
  });
  const context = await browser.newContext({
    permissions: ["microphone"],
  });
  const page = await context.newPage();

  const consoleMessages: string[] = [];
  page.on("console", (msg) => {
    consoleMessages.push(`${msg.type()}: ${msg.text()}`);
  });
  page.on("pageerror", (error) => {
    consoleMessages.push(`pageerror: ${error.stack || String(error)}`);
  });

  try {
    console.error("step: goto");
    await page.goto(pageUrl, { waitUntil: "networkidle", timeout: timeoutMs });

    const resetButton = page.getByRole("button", { name: "Reset Cached Artifacts" });
    if (await resetButton.isVisible()) {
      console.error("step: reset-cache");
      await resetButton.click();
    }

    console.error("step: download-artifacts");
    await page.getByRole("button", { name: /Download Required Files|Refresh Cache/ }).click();
    await page.waitForFunction(
      () => Array.from(document.querySelectorAll("p")).filter((el) => el.textContent?.includes("Cached locally")).length >= 5,
      undefined,
      { timeout: timeoutMs },
    );

    console.error("step: record-start");
    await page.getByRole("button", { name: /Start Recording|Record Again/ }).click();
    await page.waitForTimeout(recordMs);
    console.error("step: record-stop");
    await page.getByRole("button", { name: "Stop Recording" }).click();
    await page.waitForFunction(
      () => document.querySelector("audio") !== null,
      undefined,
      { timeout: timeoutMs },
    );

    console.error("step: run");
    await page.getByRole("button", { name: /Run FunASR|Running WebGPU Decode/ }).click();

    console.error("step: wait-result");
    await page.waitForFunction(
      () => {
        const text = document.body.textContent || "";
        return text.includes("Run completed.") || text.includes("Unsupported ONNX operation:") || text.includes("cached encoder artifact is the raw encoder graph") || text.includes("WebGPU backend is not available");
      },
      undefined,
      { timeout: timeoutMs },
    );

    const transcript = await page.locator("text=Transcript").locator("..").locator("div.font-tiktok").textContent().catch(() => null);
    const generatedIds = await readPreText(page, "Generated IDs");
    const runMetadata = await readPreText(page, "Run Metadata");
    const artifactDiagnostics = await readPreText(page, "Artifact Diagnostics");
    const progressLog = await readPreText(page, "Progress Log");
    const runtimeStatus = await page.locator("text=Runtime Status").locator("..").textContent().catch(() => null);
    console.error("step: done");

    console.log(
      JSON.stringify(
        {
          pageUrl,
          audioPath,
          transcript: transcript?.trim() || null,
          generatedIds,
          runMetadata,
          artifactDiagnostics,
          progressLog,
          runtimeStatus: runtimeStatus?.trim() || null,
          consoleMessages,
        },
        null,
        2,
      ),
    );
  } finally {
    await context.close();
    await browser.close();
  }
}

await main();

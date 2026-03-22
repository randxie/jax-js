<script lang="ts">
  import { defaultDevice, init } from "@jax-js/jax";
  import { cachedFetch, opfs, safetensors } from "@jax-js/loaders";

  import {
    fromSafetensors,
    loadTokenizer,
    MODEL_URL,
    TOKENIZER_URL,
    type PaddleOCRModel,
    runOCR,
    type Tokenizer,
  } from "./inference";

  // ── State ──────────────────────────────────────────────────
  type Phase =
    | "idle"
    | "downloading-model"
    | "downloading-tok"
    | "ready"
    | "running"
    | "done"
    | "error";

  let phase = $state<Phase>("idle");
  let downloadProgress = $state(0); // 0..1
  let downloadLabel = $state("");
  let errorMsg = $state("");
  let ocrResult = $state("");
  let ocrPartial = $state("");

  let model: PaddleOCRModel | null = null;
  let tok: Tokenizer | null = null;
  let ocrController: AbortController | null = null;

  let fileInput: HTMLInputElement;
  let previewImg: HTMLImageElement | null = $state(null);
  let previewSrc = $state<string | null>(null);

  // ── Model loading ──────────────────────────────────────────
  async function loadModel() {
    try {
      const devices = await init("webgpu");
      if (devices.includes("webgpu")) {
        defaultDevice("webgpu");
      } else {
        console.warn("WebGPU unavailable — falling back to WASM (will be slow)");
        defaultDevice("wasm");
      }

      // Download tokenizer (~2 MB)
      phase = "downloading-tok";
      downloadLabel = "Tokenizer";
      tok = await loadTokenizer(TOKENIZER_URL, (msg) => (downloadLabel = msg));

      // Download model weights (~959 MB BF16 safetensors, stored as ~1.9 GB Float32 in RAM)
      phase = "downloading-model";
      downloadLabel = "Model weights (~959 MB)";
      downloadProgress = 0;

      const data = await cachedFetch(MODEL_URL, undefined, (p) => {
        downloadProgress = p.totalBytes ? p.loadedBytes / p.totalBytes : 0;
      });

      downloadLabel = "Parsing safetensors…";
      const file = safetensors.parse(data);

      downloadLabel = "Building model…";
      model = fromSafetensors(file);

      phase = "ready";
    } catch (e: any) {
      phase = "error";
      errorMsg = String(e) + (e?.stack ? "\n\n" + e.stack : "");
      console.error("Model load error:", e);
    }
  }

  // ── Image selection ────────────────────────────────────────
  function onFileChange(e: Event) {
    const files = (e.target as HTMLInputElement).files;
    if (!files?.length) return;
    const url = URL.createObjectURL(files[0]);
    previewSrc = url;
    ocrResult = "";
    ocrPartial = "";
    const img = new Image();
    img.onload = () => (previewImg = img);
    img.src = url;
  }

  // ── OCR inference ──────────────────────────────────────────
  async function runOCRClick() {
    if (!model || !tok || !previewImg) return;
    ocrController?.abort();
    ocrController = new AbortController();
    phase = "running";
    ocrResult = "";
    ocrPartial = "";
    try {
      const result = await runOCR(model, tok, previewImg, 512, (partial) => {
        ocrPartial = partial;
      }, "ocr", ocrController.signal);
      ocrResult = result;
      ocrPartial = "";
      phase = "done";
    } catch (e: any) {
      if (e?.name === "AbortError") {
        phase = "ready";
      } else {
        phase = "error";
        errorMsg = String(e) + (e?.stack ? "\n\n" + e.stack : "");
        console.error("OCR error:", e);
      }
    } finally {
      ocrController = null;
    }
  }

  function stopOCR() {
    ocrController?.abort();
  }

  async function clearCache() {
    await opfs.clear();
    alert("Cache cleared. Reload the page to re-download.");
  }
</script>

<div class="min-h-screen bg-white">
  <header class="border-b border-gray-200">
    <div class="max-w-3xl mx-auto px-4 py-6">
      <h1 class="text-2xl font-semibold">PaddleOCR-VL-1.5</h1>
      <p class="text-gray-500 mt-1 text-sm">
        Vision-Language OCR model running in the browser via jax-js.
        <span class="font-medium">0.9 B parameters · ~959 MB BF16 weights</span>
      </p>
    </div>
  </header>

  <main class="max-w-3xl mx-auto px-4 py-8 space-y-6">

    <!-- Download section -->
    {#if phase === "idle"}
      <div class="border-2 border-black p-6">
        <h2 class="font-semibold mb-2">Step 1 — Download model</h2>
        <p class="text-sm text-gray-600 mb-4">
          The model weights are ~959 MB (BF16) and will be cached locally via the
          browser's Origin Private File System so subsequent loads are instant.
          A GPU (WebGPU) is strongly recommended; CPU fallback will be very slow.
        </p>
        <button class="btn" onclick={loadModel}> Download &amp; load model </button>
      </div>

    {:else if phase === "downloading-tok" || phase === "downloading-model"}
      <div class="border-2 border-primary p-6 bg-primary/5">
        <p class="font-medium mb-2">{downloadLabel}</p>
        {#if phase === "downloading-model"}
          <div class="w-full bg-white border border-primary/20 rounded-full h-4 overflow-hidden mb-1">
            <div
              class="bg-primary h-4 transition-all duration-100"
              style="width: {(downloadProgress * 100).toFixed(1)}%"
            ></div>
          </div>
          <p class="text-xs text-gray-500">{(downloadProgress * 100).toFixed(1)}%</p>
        {:else}
          <p class="text-sm text-gray-500 animate-pulse">Loading…</p>
        {/if}
      </div>

    {:else if phase === "ready" || phase === "running" || phase === "done"}
      <!-- Model ready -->
      <div class="flex items-center gap-2 text-sm text-green-700">
        <span class="font-bold">✓</span> Model loaded
        <button
          class="ml-auto text-xs text-gray-400 underline"
          onclick={clearCache}
        >Clear cache</button>
      </div>

      <!-- Image upload -->
      <div class="border-2 border-dashed border-gray-300 p-6 text-center">
        <p class="text-sm text-gray-500 mb-3">Upload an image containing text</p>
        <input
          type="file"
          accept="image/*"
          class="hidden"
          bind:this={fileInput}
          onchange={onFileChange}
        />
        <button class="btn" onclick={() => fileInput.click()}>
          Choose image
        </button>
      </div>

      <!-- Preview -->
      {#if previewSrc}
        <div class="flex gap-6 items-start">
          <img
            src={previewSrc}
            alt="Input image"
            class="max-w-xs max-h-64 object-contain border border-gray-200"
          />
          <div class="flex-1">
            <button
              class="btn"
              onclick={runOCRClick}
              disabled={phase === "running" || !previewImg}
            >
              {phase === "running" ? "Running OCR…" : "Run OCR"}
            </button>
            {#if phase === "running"}
              <button class="btn mt-2" onclick={stopOCR}>
                Stop
              </button>
            {/if}

            <!-- Streaming partial result -->
            {#if ocrPartial}
              <div class="mt-4 p-3 bg-gray-50 border border-gray-200 rounded text-sm font-mono whitespace-pre-wrap text-gray-600">
                {ocrPartial}<span class="animate-pulse">▌</span>
              </div>
            {/if}

            <!-- Final result -->
            {#if ocrResult}
              <div class="mt-4">
                <p class="text-xs uppercase tracking-wide text-gray-400 mb-1">OCR Output</p>
                <pre
                  class="p-3 bg-gray-50 border border-gray-200 rounded text-sm font-mono whitespace-pre-wrap overflow-auto max-h-80"
                >{ocrResult}</pre>
                <button
                  class="mt-2 text-xs underline text-gray-400"
                  onclick={() => navigator.clipboard.writeText(ocrResult)}
                >
                  Copy to clipboard
                </button>
              </div>
            {/if}
          </div>
        </div>
      {/if}

    {:else if phase === "error"}
      <div class="border-2 border-red-400 bg-red-50 p-4 text-sm text-red-700">
        <strong>Error:</strong>
        <pre class="mt-2 whitespace-pre-wrap font-mono text-xs overflow-auto max-h-60">{errorMsg}</pre>
        <button class="mt-2 block underline" onclick={() => (phase = "idle")}>
          Try again
        </button>
      </div>
    {/if}

    <!-- Implementation notes -->
    <details class="text-sm text-gray-500">
      <summary class="cursor-pointer font-medium text-gray-700">
        Implementation details
      </summary>
      <div class="mt-3 space-y-2 pl-2 border-l-2 border-gray-100">
        <p>
          <strong>BF16 loading</strong> — BF16 (bfloat16) weights are converted to Float32 in
          <code>packages/loaders/src/safetensors.ts</code> by reinterpreting each uint16 as the
          upper 16 bits of a Float32 (a zero-cost bit shift: <code>int32 = uint16 &lt;&lt; 16</code>).
        </p>
        <p>
          <strong>RMSNorm</strong> — implemented as
          <code>x / sqrt(mean(x²) + ε) * weight</code> (no mean subtraction, no bias),
          as used throughout the LLM text decoder.
        </p>
        <p>
          <strong>3D Multimodal RoPE</strong> — the 128-dim head is split into three frequency
          bands [16, 24, 24] for (temporal, height, width) positions. Image tokens receive
          2D spatial coordinates; text tokens receive equal 1D coordinates on all three axes.
        </p>
        <p>
          <strong>Grouped Query Attention (GQA)</strong> — 16 query heads share only 2 KV heads
          (group size 8). Implemented naively with <code>np.repeat(k, 8, axis=1)</code> before
          the attention kernel.
        </p>
        <p>
          <strong>SiLU gated MLP</strong> — <code>silu(gate_proj(x)) * up_proj(x)</code> then
          <code>down_proj</code>, used in all text decoder layers.
        </p>
        <p>
          <strong>Patch embedding</strong> — implemented via im2col reshape + matmul rather than
          conv2d, since the patch size divides evenly into the image size.
        </p>
      </div>
    </details>
  </main>
</div>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-5 py-2.5 border-2 border-black;
    @apply enabled:hover:bg-black enabled:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
  }
</style>

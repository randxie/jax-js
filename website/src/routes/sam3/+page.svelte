<script lang="ts">
  import { blockUntilReady, defaultDevice, init, jit } from "@jax-js/jax";
  import { ONNXModel } from "@jax-js/onnx";

  import {
    type ImageFeatures,
    type LanguageFeatures,
    type SAM3Prediction,
    preprocessImage,
    promptTokensToArray,
    renderMasks,
    runDecoder,
    runLanguageEncoder,
    tokenizePrompt,
  } from "./sam3";

  type ONNXModelInstance = InstanceType<typeof ONNXModel>;
  type ModelRunner = (inputs: Record<string, any>) => any;

  const EXAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Afrykarium_tunel.jpg/1280px-Afrykarium_tunel.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/0/00/Gats_domestics.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Desk333.JPG/1280px-Desk333.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg/1280px-Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg",
  ];

  let canvasEl: HTMLCanvasElement;
  let imgEl: HTMLImageElement;

  let imageEncoderModel = $state<ONNXModelInstance | null>(null);
  let languageEncoderModel = $state<ONNXModelInstance | null>(null);
  let decoderModel = $state<ONNXModelInstance | null>(null);

  let runImageEncoderJit: ModelRunner | null = null;

  // Model file state
  type ModelFile = { name: string; sizeGB: string } | null;
  let imgEncFile = $state<ModelFile>(null);
  let langEncFile = $state<ModelFile>(null);
  let decFile = $state<ModelFile>(null);

  // Cached image features
  let cachedImageFeatures: ImageFeatures | null = null;
  let cachedImageUrl = "";

  let textPrompt = $state("a cat");
  let currentImageUrl = $state(EXAMPLE_IMAGES[0]);
  let imageIndex = $state(0);
  let status = $state("");
  let isRunning = $state(false);
  let isLoadingModels = $state(false);
  let predictions = $state<SAM3Prediction | null>(null);
  let scoreThreshold = $state(0.5);

  const allModelsLoaded = $derived(
    imageEncoderModel !== null &&
      languageEncoderModel !== null &&
      decoderModel !== null,
  );

  async function readFile(file: File): Promise<Uint8Array> {
    const buf = await file.arrayBuffer();
    return new Uint8Array(buf);
  }

  async function pickImageEncoder(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    imgEncFile = { name: file.name, sizeGB: (file.size / 1e9).toFixed(2) };
    status = `Loading image encoder (${imgEncFile.sizeGB} GB)…`;
    isLoadingModels = true;
    try {
      const bytes = await readFile(file);
      await ensureInit();
      imageEncoderModel = new ONNXModel(bytes);
      runImageEncoderJit = jit((inputs: Record<string, any>) =>
        imageEncoderModel!.run(inputs),
      );
      cachedImageFeatures = null;
      status = "Image encoder ready.";
    } catch (err) {
      status = `Error loading image encoder: ${err instanceof Error ? err.message : err}`;
      imageEncoderModel = null;
    } finally {
      isLoadingModels = false;
    }
  }

  async function pickLanguageEncoder(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    langEncFile = { name: file.name, sizeGB: (file.size / 1e9).toFixed(2) };
    status = `Loading language encoder (${langEncFile.sizeGB} GB)…`;
    isLoadingModels = true;
    try {
      const bytes = await readFile(file);
      await ensureInit();
      languageEncoderModel = new ONNXModel(bytes);
      status = "Language encoder ready.";
    } catch (err) {
      status = `Error loading language encoder: ${err instanceof Error ? err.message : err}`;
      languageEncoderModel = null;
    } finally {
      isLoadingModels = false;
    }
  }

  async function pickDecoder(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    decFile = { name: file.name, sizeGB: (file.size / 1e6).toFixed(0) + " MB" };
    status = `Loading decoder (${decFile.sizeGB})…`;
    isLoadingModels = true;
    try {
      const bytes = await readFile(file);
      await ensureInit();
      decoderModel = new ONNXModel(bytes);
      status = "Decoder ready.";
    } catch (err) {
      status = `Error loading decoder: ${err instanceof Error ? err.message : err}`;
      decoderModel = null;
    } finally {
      isLoadingModels = false;
    }
  }

  let _initialized = false;
  async function ensureInit() {
    if (_initialized) return;
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      throw new Error("WebGPU is not supported on this device/browser.");
    }
    defaultDevice("webgpu");
    _initialized = true;
  }

  async function loadImageToCanvas(url: string) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = reject;
      img.src = url;
    });
    canvasEl.width = img.naturalWidth;
    canvasEl.height = img.naturalHeight;
    const ctx = canvasEl.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    return { width: img.naturalWidth, height: img.naturalHeight, ctx, img };
  }

  function redrawCanvas() {
    if (!imgEl?.complete) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.drawImage(imgEl, 0, 0, canvasEl.width, canvasEl.height);
    if (predictions) {
      renderMasks(canvasEl, predictions, scoreThreshold);
    }
  }

  async function runInference() {
    if (isRunning || !allModelsLoaded) return;
    isRunning = true;
    predictions = null;

    try {
      const { width, height, ctx, img } =
        await loadImageToCanvas(currentImageUrl);

      if (cachedImageUrl !== currentImageUrl || cachedImageFeatures === null) {
        status = "Encoding image… (first run compiles shaders, may take ~30s)";
        const imageRgb = preprocessImage(ctx, width, height);
        const raw = runImageEncoderJit!({ image: imageRgb });
        await blockUntilReady(raw);
        cachedImageFeatures = raw as ImageFeatures;
        cachedImageUrl = currentImageUrl;
      }

      status = "Encoding text prompt…";
      const tokenArray = await tokenizePrompt(textPrompt);
      const tokens = promptTokensToArray(tokenArray);
      const langFeats = runLanguageEncoder(
        languageEncoderModel!,
        tokens,
      ) as LanguageFeatures;
      await blockUntilReady(langFeats as any);

      status = "Decoding masks…";
      const raw = runDecoder(
        decoderModel!,
        cachedImageFeatures!,
        langFeats,
        height,
        width,
      );
      await blockUntilReady(raw as any);

      const boxes = (await raw.boxes.jsAsync()) as Float32Array;
      const scores = (await raw.scores.jsAsync()) as Float32Array;
      const masksArr = (await raw.masks.jsAsync()) as Uint8Array;

      predictions = {
        boxes,
        scores,
        masks: masksArr,
        maskShape: raw.masks.shape,
      };
      imgEl = img;
      renderMasks(canvasEl, predictions, scoreThreshold);

      status = `Found ${scores.filter((s) => s >= scoreThreshold).length} instance(s) above threshold.`;
    } catch (e) {
      status = `Error: ${e instanceof Error ? e.message : String(e)}`;
      console.error(e);
    } finally {
      isRunning = false;
    }
  }

  function nextImage() {
    imageIndex = (imageIndex + 1) % EXAMPLE_IMAGES.length;
    currentImageUrl = EXAMPLE_IMAGES[imageIndex];
    cachedImageFeatures = null;
    predictions = null;
    status = "";
  }

  $effect(() => {
    scoreThreshold;
    predictions;
    redrawCanvas();
  });
</script>

<!-- Hidden image used for canvas re-draw -->
<img
  bind:this={imgEl}
  src={currentImageUrl}
  crossorigin="anonymous"
  class="hidden"
  alt=""
/>

<main class="p-4 max-w-3xl mx-auto">
  <h1 class="text-xl font-bold mb-1">SAM 3 — Segment Anything with Concepts</h1>
  <p class="text-sm text-gray-600 mb-4">
    Open-vocabulary segmentation: describe what you want to find, and SAM 3
    segments it. Runs locally in your browser via WebGPU — no data leaves your
    machine.
  </p>

  <!-- ── Model loading ─────────────────────────────────────────────────────── -->
  <section
    class="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4 text-sm"
  >
    <p class="font-medium mb-2">Load local model files</p>
    <p class="text-gray-600 mb-3 text-xs">
      Generate merged ONNX files with
      <code class="bg-amber-100 px-1 rounded"
        >python scripts/sam3_merge_onnx.py</code
      >. Then pick the three files from
      <code class="bg-amber-100 px-1 rounded">/tmp/sam3_merged/</code>.
    </p>
    <div class="grid gap-2">
      {#snippet fileRow(
        label: string,
        hint: string,
        loaded: boolean,
        file: ModelFile,
        onchange: (e: Event) => void,
      )}
        {@const disabled = isLoadingModels || isRunning}
        <div class="flex items-center gap-3">
          <span class="w-36 shrink-0 text-gray-700 text-sm">{label}</span>
          <span class="text-xs text-gray-400 w-20 shrink-0">{hint}</span>
          <span
            class="text-xs px-2 py-0.5 rounded-full shrink-0"
            class:bg-green-100={loaded}
            class:text-green-700={loaded}
            class:bg-gray-100={!loaded}
            class:text-gray-500={!loaded}
          >
            {loaded ? "✓ loaded" : "not loaded"}
          </span>
          {#if file}
            <span class="text-xs text-gray-500 truncate">{file.name}</span>
          {/if}
          <label
            class="ml-auto text-xs border rounded px-2 py-0.5 cursor-pointer"
            class:opacity-50={disabled}
            class:pointer-events-none={disabled}
          >
            Browse…
            <input
              type="file"
              accept=".onnx"
              {onchange}
              class="sr-only"
              {disabled}
            />
          </label>
        </div>
      {/snippet}
      {@render fileRow(
        "Image encoder",
        "~1.8 GB",
        imageEncoderModel !== null,
        imgEncFile,
        pickImageEncoder,
      )}
      {@render fileRow(
        "Language encoder",
        "~1.6 GB",
        languageEncoderModel !== null,
        langEncFile,
        pickLanguageEncoder,
      )}
      {@render fileRow(
        "Decoder",
        "~130 MB",
        decoderModel !== null,
        decFile,
        pickDecoder,
      )}
    </div>
  </section>

  <!-- ── Inference controls ────────────────────────────────────────────────── -->
  <div class="flex flex-col gap-3 mb-4">
    <div class="flex gap-2 items-center flex-wrap">
      <label
        class="text-sm font-medium whitespace-nowrap"
        for="sam3-text-prompt">Text prompt:</label
      >
      <input
        id="sam3-text-prompt"
        bind:value={textPrompt}
        type="text"
        class="border rounded px-2 py-1 text-sm flex-1 min-w-40"
        placeholder="e.g. a cat, a person, a chair"
        onkeydown={(e) =>
          e.key === "Enter" && !isRunning && allModelsLoaded && runInference()}
      />
    </div>

    <div class="flex gap-2 flex-wrap items-center">
      <button
        onclick={runInference}
        disabled={isRunning || !allModelsLoaded || isLoadingModels}
        class="btn-primary"
        title={!allModelsLoaded ? "Load all three model files first" : ""}
      >
        {isRunning ? "Running…" : "Segment"}
      </button>
      <button onclick={nextImage} disabled={isRunning} class="btn-secondary">
        Next image
      </button>
      <label class="text-sm ml-2">
        Threshold:
        <input
          type="range"
          min="0.1"
          max="0.95"
          step="0.05"
          bind:value={scoreThreshold}
          class="ml-1"
        />
        <span class="tabular-nums">{scoreThreshold.toFixed(2)}</span>
      </label>
    </div>
  </div>

  {#if status}
    <p class="text-sm text-gray-600 mb-2">{status}</p>
  {/if}

  {#if predictions}
    <p class="text-sm text-gray-500 mb-2">
      {predictions.scores.length} detections,
      {predictions.scores.filter((s) => s >= scoreThreshold).length} above threshold
    </p>
  {/if}

  <div class="-mx-4 sm:mx-0">
    <canvas
      bind:this={canvasEl}
      class="max-w-full border border-gray-200 rounded"
    ></canvas>
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  .btn-primary {
    @apply border rounded px-3 py-1 text-sm bg-blue-600 text-white hover:bg-blue-700 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed;
  }
  .btn-secondary {
    @apply border rounded px-3 py-1 text-sm hover:bg-gray-100 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed;
  }
</style>

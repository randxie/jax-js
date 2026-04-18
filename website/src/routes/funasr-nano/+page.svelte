<script lang="ts">
  import { defaultDevice, init, numpy as np } from "@jax-js/jax";
  import { tokenizers } from "@jax-js/loaders";
  import { ONNXModel } from "@jax-js/onnx";
  import { AudioLinesIcon, CpuIcon, FileUpIcon, SparklesIcon } from "@lucide/svelte";

  import Seo from "$lib/common/Seo.svelte";
  import { extractFunASRNanoSpeech } from "../../../../scripts/funasr_nano_frontend";
  import { prepareFunASRNanoSourceIds } from "../../../../scripts/funasr_nano_prepare";
  import {
    buildFunASRNanoInputEmbeds,
    decodeGeneratedText,
    generateFunASRNanoGreedy,
    loadFunASRNanoQwenFromBuffers,
  } from "../../../../scripts/funasr_nano_qwen";

  type RequiredFileKey =
    | "sample"
    | "encoder"
    | "llm"
    | "tokenizer"
    | "config"
    | "generationConfig";

  const fileLabels: Record<RequiredFileKey, string> = {
    sample: "funasr_nano_encoder_sample.json",
    encoder: "funasr_nano_encoder.onnx",
    llm: "funasr_nano_llm_fp16.safetensors",
    tokenizer: "Qwen tokenizer.json",
    config: "Qwen config.json",
    generationConfig: "Qwen generation_config.json",
  };

  let files = $state<Partial<Record<RequiredFileKey, File>>>({});
  let status = $state<"idle" | "running" | "done" | "error">("idle");
  let errorMessage = $state<string | null>(null);
  let transcript = $state<string>("");
  let generatedIds = $state<number[]>([]);
  let runMeta = $state<Record<string, unknown> | null>(null);
  let maxNewTokens = $state(16);
  const webgpuAvailable = $derived(typeof navigator !== "undefined" && !!navigator.gpu);

  function setFile(key: RequiredFileKey, fileList: FileList | null) {
    files[key] = fileList?.[0];
  }

  function missingFiles(): string[] {
    return (Object.keys(fileLabels) as RequiredFileKey[])
      .filter((key) => !files[key])
      .map((key) => fileLabels[key]);
  }

  async function readBytes(file: File): Promise<Uint8Array<ArrayBuffer>> {
    return new Uint8Array(await file.arrayBuffer());
  }

  async function run() {
    const missing = missingFiles();
    if (missing.length > 0) {
      status = "error";
      errorMessage = `Missing files: ${missing.join(", ")}`;
      return;
    }

    status = "running";
    errorMessage = null;
    transcript = "";
    generatedIds = [];
    runMeta = null;

    try {
      const devices = await init("webgpu");
      if (!devices.includes("webgpu")) {
        throw new Error("WebGPU backend is not available in this browser");
      }
      defaultDevice("webgpu");

      const sample = JSON.parse(await files.sample!.text());
      const waveform = new Float32Array(sample.waveform);
      const { speech, shape } = await extractFunASRNanoSpeech(
        waveform,
        sample.frontend,
      );

      const encoder = new ONNXModel(await readBytes(files.encoder!));
      const speechLengths = np.array(new Int32Array([shape[1]]), {
        dtype: np.int32,
        shape: [1],
      });
      const encoderOut = encoder.run({
        speech: np.array(speech, { shape }),
        speech_lengths: speechLengths,
      }).encoder_out;

      const tokenizer = tokenizers.HuggingFaceBPE.fromBinary(
        await readBytes(files.tokenizer!),
      );
      const prepared = prepareFunASRNanoSourceIds(
        tokenizer,
        ".download/Fun-ASR-Nano-2512/example/zh.mp3",
        shape[1],
      );

      const qwen = loadFunASRNanoQwenFromBuffers(
        await readBytes(files.llm!),
        await files.config!.text(),
        await files.generationConfig!.text(),
      );
      const baseEmbeds = await buildFunASRNanoInputEmbeds(
        qwen,
        prepared.sourceIds,
        prepared.fbankBeg,
        prepared.fakeTokenLen,
        encoderOut,
      );

      const ids = await generateFunASRNanoGreedy(qwen, baseEmbeds, maxNewTokens);
      generatedIds = ids;
      transcript = decodeGeneratedText(tokenizer, ids);
      runMeta = {
        speech_shape: shape,
        fbank_beg: prepared.fbankBeg,
        fake_token_len: prepared.fakeTokenLen,
      };
      status = "done";
    } catch (error) {
      console.error(error);
      status = "error";
      errorMessage = error instanceof Error ? error.message : String(error);
    }
  }
</script>

<Seo
  title="FunASR Nano WebGPU Runner"
  description="Browser-side WebGPU staging page for FunASR-nano in jax-js."
/>

<svelte:head>
  <title>FunASR Nano WebGPU Runner</title>
</svelte:head>

<main class="min-h-screen bg-[radial-gradient(circle_at_top,#f4f0d6_0%,#f7f7f2_38%,#ecefe5_100%)] text-stone-900">
  <section class="mx-auto max-w-6xl px-6 py-10">
    <div class="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
      <div class="rounded-[2rem] border border-stone-300/60 bg-white/80 p-8 shadow-[0_20px_60px_rgba(64,64,32,0.08)] backdrop-blur">
        <div class="mb-6 flex items-center gap-3">
          <div class="rounded-2xl bg-lime-200 p-3 text-stone-900">
            <AudioLinesIcon size={24} />
          </div>
          <div>
            <p class="font-tiktok text-sm uppercase tracking-[0.28em] text-stone-500">
              WebGPU Staging
            </p>
            <h1 class="font-tiktok text-4xl leading-tight">FunASR Nano Runner</h1>
          </div>
        </div>

        <p class="max-w-2xl text-base leading-7 text-stone-600">
          This page is the browser-side WebGPU harness for the current FunASR-nano
          work. It runs from exported local artifacts instead of bundling 1GB+ model
          files into the repo. The current input is the exported sample JSON, not raw
          MP3 decode in-browser.
        </p>

        <div class="mt-8 grid gap-4 sm:grid-cols-2">
          {#each Object.entries(fileLabels) as [key, label]}
            <label class="rounded-2xl border border-stone-300 bg-stone-50/90 p-4 transition hover:border-lime-500 hover:bg-white">
              <div class="mb-2 flex items-center gap-2 text-sm font-medium text-stone-700">
                <FileUpIcon size={16} />
                {label}
              </div>
              <input
                class="block w-full text-sm text-stone-600 file:mr-3 file:rounded-full file:border-0 file:bg-lime-700 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-lime-800"
                type="file"
                onchange={(event) =>
                  setFile(key as RequiredFileKey, (event.currentTarget as HTMLInputElement).files)}
              />
              {#if files[key as RequiredFileKey]}
                <p class="mt-2 truncate text-xs text-stone-500">
                  {files[key as RequiredFileKey]?.name}
                </p>
              {/if}
            </label>
          {/each}
        </div>

        <div class="mt-6 flex flex-wrap items-end gap-4">
          <label class="flex flex-col gap-2 text-sm text-stone-600">
            Max new tokens
            <input
              class="w-28 rounded-xl border border-stone-300 bg-white px-3 py-2 text-stone-900"
              type="number"
              min="1"
              max="64"
              bind:value={maxNewTokens}
            />
          </label>

          <button
            class="inline-flex items-center gap-2 rounded-full bg-stone-900 px-6 py-3 font-medium text-white transition hover:bg-lime-800 disabled:cursor-not-allowed disabled:bg-stone-400"
            disabled={status === "running" || !webgpuAvailable}
            onclick={run}
          >
            <SparklesIcon size={18} />
            {status === "running" ? "Running WebGPU Decode…" : "Run FunASR"}
          </button>
        </div>
      </div>

      <div class="flex flex-col gap-6">
        <div class="rounded-[2rem] border border-stone-300/60 bg-stone-950 p-6 text-stone-100 shadow-[0_20px_60px_rgba(20,20,12,0.18)]">
          <div class="mb-4 flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-lime-300">
            <CpuIcon size={16} />
            Runtime Status
          </div>

          {#if !webgpuAvailable}
            <p class="text-sm leading-7 text-stone-300">
              `navigator.gpu` is not available in this browser, so this page cannot
              initialize the `webgpu` backend.
            </p>
          {:else if status === "idle"}
            <p class="text-sm leading-7 text-stone-300">
              Load the exported FunASR artifacts and run the browser-side WebGPU path.
            </p>
          {:else if status === "running"}
            <p class="text-sm leading-7 text-lime-200">
              WebGPU backend initialized. Frontend, ONNX encoder/adaptor, and Qwen
              decode are running in-browser.
            </p>
          {:else if status === "done"}
            <p class="text-sm leading-7 text-lime-200">
              Run completed. Transcript and generated token IDs are shown below.
            </p>
          {:else if status === "error"}
            <p class="text-sm leading-7 text-rose-300">
              {errorMessage}
            </p>
          {/if}
        </div>

        <div class="rounded-[2rem] border border-stone-300/60 bg-white/90 p-6 shadow-[0_16px_50px_rgba(64,64,32,0.08)]">
          <p class="mb-3 text-sm uppercase tracking-[0.22em] text-stone-500">
            Transcript
          </p>
          <div class="min-h-28 rounded-2xl bg-stone-100 p-4 font-tiktok text-xl leading-relaxed text-stone-900">
            {transcript || "No transcript yet."}
          </div>
        </div>

        <div class="rounded-[2rem] border border-stone-300/60 bg-white/90 p-6 shadow-[0_16px_50px_rgba(64,64,32,0.08)]">
          <p class="mb-3 text-sm uppercase tracking-[0.22em] text-stone-500">
            Generated IDs
          </p>
          <pre class="overflow-x-auto rounded-2xl bg-stone-100 p-4 text-sm leading-6 text-stone-700">{generatedIds.length > 0 ? JSON.stringify(generatedIds, null, 2) : "[]"}</pre>
        </div>

        <div class="rounded-[2rem] border border-stone-300/60 bg-white/90 p-6 shadow-[0_16px_50px_rgba(64,64,32,0.08)]">
          <p class="mb-3 text-sm uppercase tracking-[0.22em] text-stone-500">
            Run Metadata
          </p>
          <pre class="overflow-x-auto rounded-2xl bg-stone-100 p-4 text-sm leading-6 text-stone-700">{runMeta ? JSON.stringify(runMeta, null, 2) : "{}"}</pre>
        </div>
      </div>
    </div>
  </section>
</main>

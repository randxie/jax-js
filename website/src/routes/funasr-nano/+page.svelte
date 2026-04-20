<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { defaultDevice, init, numpy as np } from "@jax-js/jax";
  import { cachedFetch, opfs, tokenizers } from "@jax-js/loaders";
  import { ONNXModel } from "@jax-js/onnx";
  import {
    AudioLinesIcon,
    CpuIcon,
    DownloadIcon,
    FileUpIcon,
    MicIcon,
    SquareIcon,
    SparklesIcon,
  } from "@lucide/svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import Seo from "$lib/common/Seo.svelte";
  import { extractFunASRNanoSpeech } from "../../../../scripts/funasr_nano_frontend";
  import { prepareFunASRNanoSourceIds } from "../../../../scripts/funasr_nano_prepare";
  import {
    buildFunASRNanoInputEmbeds,
    decodeGeneratedText,
    generateFunASRNanoGreedy,
    loadFunASRNanoQwenFromBuffers,
  } from "../../../../scripts/funasr_nano_qwen";

  type ArtifactKey =
    | "encoder"
    | "llm"
    | "tokenizer"
    | "config"
    | "generationConfig";

  const fileLabels: Record<ArtifactKey, string> = {
    encoder: "funasr_nano_encoder.onnx",
    llm: "funasr_nano_llm_fp16.safetensors",
    tokenizer: "Qwen tokenizer.json",
    config: "Qwen config.json",
    generationConfig: "Qwen generation_config.json",
  };
  const defaultArtifactUrls: Record<ArtifactKey, string> = {
    encoder:
      "https://huggingface.co/csukuangfj/sherpa-onnx-funasr-nano-2025-12-30/resolve/main/encoder_adaptor.onnx",
    llm:
      "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/6a902ecfb2da967c517fdc0b5cad4ece88d22c18/Qwen3-0.6B/model.safetensors",
    tokenizer:
      "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/6a902ecfb2da967c517fdc0b5cad4ece88d22c18/Qwen3-0.6B/tokenizer.json",
    config:
      "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/6a902ecfb2da967c517fdc0b5cad4ece88d22c18/Qwen3-0.6B/config.json",
    generationConfig:
      "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/6a902ecfb2da967c517fdc0b5cad4ece88d22c18/Qwen3-0.6B/generation_config.json",
  };

  const frontendConfig = {
    fs: 16000,
    window: "hamming",
    n_mels: 80,
    frame_length: 25,
    frame_shift: 10,
    lfr_m: 7,
    lfr_n: 6,
    upsacle_samples: true,
  } as const;
  const encoderFrames = 94;
  const encoderFeatureDim = frontendConfig.n_mels * frontendConfig.lfr_m;

  let artifactUrls = $state<Record<ArtifactKey, string>>({ ...defaultArtifactUrls });
  let artifactCache = $state<Partial<Record<ArtifactKey, { cached: boolean; size: number }>>>({});
  let status = $state<"idle" | "running" | "done" | "error">("idle");
  let errorMessage = $state<string | null>(null);
  let transcript = $state<string>("");
  let generatedIds = $state<number[]>([]);
  let runMeta = $state<Record<string, unknown> | null>(null);
  let artifactDiagnostics = $state<Record<string, unknown> | null>(null);
  let maxNewTokens = $state(16);
  let artifactsReady = $state(false);
  let isPreparingArtifacts = $state(false);
  let recordingState = $state<"idle" | "recording" | "ready">("idle");
  let recordedAudio = $state<File | null>(null);
  let recordedAudioUrl = $state<string | null>(null);
  let audioInputDevices = $state<MediaDeviceInfo[]>([]);
  let selectedInputDeviceId = $state<string>("");
  let inputDeviceLabel = $state<string | null>(null);
  let inputDeviceId = $state<string | null>(null);
  let outputDeviceLabel = $state<string | null>(null);
  const webgpuAvailable = $derived(typeof navigator !== "undefined" && !!navigator.gpu);
  let mediaStream: MediaStream | null = null;
  let mediaRecorder: MediaRecorder | null = null;
  let recordingChunks: Blob[] = [];
  let downloadManager: DownloadManager;

  function missingArtifactUrls(): string[] {
    return (Object.keys(fileLabels) as ArtifactKey[])
      .filter((key) => artifactUrls[key].trim() === "")
      .map((key) => fileLabels[key]);
  }

  function clearRecordedAudio() {
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
      recordedAudioUrl = null;
    }
    recordedAudio = null;
    if (recordingState !== "recording") {
      recordingState = "idle";
    }
  }

  async function refreshAudioOutputLabel() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      outputDeviceLabel = "System default audio output";
      return;
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const defaultOutput = devices.find(
      (device) => device.kind === "audiooutput" && device.deviceId === "default",
    );
    const firstOutput = devices.find((device) => device.kind === "audiooutput");
    outputDeviceLabel =
      defaultOutput?.label || firstOutput?.label || "System default audio output";
  }

  async function refreshAudioInputDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      audioInputDevices = [];
      return;
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    audioInputDevices = inputs;
    if (!selectedInputDeviceId && inputs.length > 0) {
      selectedInputDeviceId = inputs[0].deviceId;
    }
  }

  function getEncoderOutputName(outputNames: string[]): string {
    return outputNames.find((name) => name === "encoder_out") || outputNames[0] || "";
  }

  function runEncoderCompat(
    encoder: ONNXModel,
    speech: Float32Array,
    shape: [number, number, number],
    speechLengths: np.Array,
  ) {
    const inputNames = encoder.model.graph?.input.map((value) => value.name) ?? [];
    const outputNames = encoder.model.graph?.output.map((value) => value.name) ?? [];
    const speechArray = np.array(speech, { shape });
    const encoderOutputName = getEncoderOutputName(outputNames);

    if (!encoderOutputName) {
      throw new Error("Encoder model does not expose any outputs");
    }

    if (inputNames.includes("speech")) {
      const out = encoder.run({
        speech: speechArray,
        ...(inputNames.includes("speech_lengths") ? { speech_lengths: speechLengths } : {}),
      });
      return {
        encoderOut: out[encoderOutputName],
        inputNames,
        outputNames,
      };
    }

    if (inputNames.includes("x")) {
      const out = encoder.run({
        x: speechArray,
        ...(inputNames.includes("input_lengths") ? { input_lengths: speechLengths } : {}),
        ...(inputNames.includes("x_length") ? { x_length: speechLengths } : {}),
        ...(inputNames.includes("speech_lengths") ? { speech_lengths: speechLengths } : {}),
      });
      return {
        encoderOut: out[encoderOutputName],
        inputNames,
        outputNames,
      };
    }

    throw new Error(`Unsupported encoder inputs: ${inputNames.join(", ") || "(none)"}`);
  }

  function stopMediaStream() {
    if (!mediaStream) return;
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
    mediaStream = null;
  }

  onDestroy(() => {
    stopMediaStream();
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
  });

  async function refreshArtifactCache() {
    const nextCache: Partial<Record<ArtifactKey, { cached: boolean; size: number }>> = {};
    for (const key of Object.keys(fileLabels) as ArtifactKey[]) {
      const url = artifactUrls[key].trim();
      if (!url) continue;
      const info = await opfs.info(url);
      nextCache[key] = {
        cached: info !== null,
        size: info?.size ?? 0,
      };
    }
    artifactCache = nextCache;
    artifactsReady = (Object.keys(fileLabels) as ArtifactKey[]).every(
      (key) => artifactUrls[key].trim() !== "" && nextCache[key]?.cached,
    );
    artifactDiagnostics = {
      ...(artifactDiagnostics ?? {}),
      configured_urls: { ...artifactUrls },
      cache: Object.fromEntries(
        (Object.keys(fileLabels) as ArtifactKey[]).map((key) => [
          key,
          {
            cached: nextCache[key]?.cached ?? false,
            size_mb: Math.round((nextCache[key]?.size ?? 0) / 1024 / 1024),
          },
        ]),
      ),
    };
  }

  async function readArtifact(key: ArtifactKey): Promise<Uint8Array<ArrayBuffer>> {
    const url = artifactUrls[key].trim();
    if (!url) {
      throw new Error(`Missing URL for ${fileLabels[key]}`);
    }
    return cachedFetch(url);
  }

  async function prepareArtifacts() {
    const missing = missingArtifactUrls();
    if (missing.length > 0) {
      status = "error";
      errorMessage = `Missing artifact URLs: ${missing.join(", ")}`;
      return;
    }

    isPreparingArtifacts = true;
    errorMessage = null;
    try {
      for (const key of Object.keys(fileLabels) as ArtifactKey[]) {
        const url = artifactUrls[key].trim();
        const cached = await opfs.info(url);
        if (!cached) {
          await downloadManager.fetch(fileLabels[key], url);
        }
      }
      await refreshArtifactCache();
    } catch (error) {
      status = "error";
      errorMessage = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      isPreparingArtifacts = false;
    }
  }

  function validateEncoderUrl(url: string) {
    if (url.includes("Fun-ASR-Nano-2512-Encoder-ONNX-FP32/resolve/main/model.onnx")) {
      throw new Error(
        "The configured encoder URL points to the raw 512-dim encoder. Use encoder_adaptor.onnx instead.",
      );
    }
  }

  async function resetArtifacts() {
    isPreparingArtifacts = true;
    errorMessage = null;
    try {
      for (const key of Object.keys(fileLabels) as ArtifactKey[]) {
        const url = artifactUrls[key].trim();
        if (!url) continue;
        await opfs.remove(url);
      }
      artifactDiagnostics = null;
      await refreshArtifactCache();
    } catch (error) {
      status = "error";
      errorMessage = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      isPreparingArtifacts = false;
    }
  }

  async function decodeBrowserAudio(
    file: File,
    sampleRate: number,
  ): Promise<Float32Array> {
    const audioContext = new AudioContext();
    try {
      const decoded = await audioContext.decodeAudioData(await file.arrayBuffer());
      let mono = decoded;
      if (decoded.numberOfChannels > 1) {
        const mixed = new AudioBuffer({
          length: decoded.length,
          numberOfChannels: 1,
          sampleRate: decoded.sampleRate,
        });
        const out = mixed.getChannelData(0);
        for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
          const data = decoded.getChannelData(channel);
          for (let i = 0; i < data.length; i++) {
            out[i] += data[i] / decoded.numberOfChannels;
          }
        }
        mono = mixed;
      }

      if (mono.sampleRate === sampleRate) {
        return mono.getChannelData(0).slice();
      }

      const offline = new OfflineAudioContext(1, Math.ceil(mono.duration * sampleRate), sampleRate);
      const source = offline.createBufferSource();
      source.buffer = mono;
      source.connect(offline.destination);
      source.start();
      const rendered = await offline.startRendering();
      return rendered.getChannelData(0).slice();
    } finally {
      await audioContext.close();
    }
  }

  function trimWaveformSilence(
    waveform: Float32Array,
    threshold = 0.003,
    paddingSamples = Math.floor(frontendConfig.fs * 0.1),
  ): Float32Array {
    let start = 0;
    while (start < waveform.length && Math.abs(waveform[start]) < threshold) {
      start++;
    }

    let end = waveform.length - 1;
    while (end >= 0 && Math.abs(waveform[end]) < threshold) {
      end--;
    }

    if (start > end) {
      return waveform;
    }

    const trimmedStart = Math.max(0, start - paddingSamples);
    const trimmedEnd = Math.min(waveform.length, end + paddingSamples + 1);
    return waveform.slice(trimmedStart, trimmedEnd);
  }

  function normalizeSpeechFrames(
    speech: Float32Array,
    shape: [number, number, number],
    targetFrames: number,
  ): { speech: Float32Array; shape: [number, number, number] } {
    const frameCount = shape[1];
    const featureDim = shape[2];
    if (featureDim !== encoderFeatureDim) {
      throw new Error(`Unexpected feature dim: ${featureDim}`);
    }
    if (frameCount === targetFrames) {
      return { speech, shape };
    }

    const normalized = new Float32Array(targetFrames * featureDim);
    const copyFrames = Math.min(frameCount, targetFrames);
    normalized.set(speech.subarray(0, copyFrames * featureDim));
    return {
      speech: normalized,
      shape: [1, targetFrames, featureDim],
    };
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      status = "error";
      errorMessage = "Microphone recording is not available in this browser";
      return;
    }

    try {
      clearRecordedAudio();
      errorMessage = null;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          ...(selectedInputDeviceId
            ? { deviceId: { exact: selectedInputDeviceId } }
            : {}),
          channelCount: 1,
          noiseSuppression: false,
          echoCancellation: false,
          autoGainControl: false,
        },
      });
      await refreshAudioInputDevices();
      const track = mediaStream.getAudioTracks()[0];
      inputDeviceLabel = track?.label || "Unnamed microphone";
      inputDeviceId = track?.getSettings().deviceId || null;
      await refreshAudioOutputLabel();
      mediaRecorder = mimeType
        ? new MediaRecorder(mediaStream, { mimeType })
        : new MediaRecorder(mediaStream);
      recordingChunks = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunks.push(event.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(
          recordingChunks,
          { type: mediaRecorder?.mimeType || "audio/webm" },
        );
        recordedAudio = new File([blob], "microphone.webm", { type: blob.type });
        recordedAudioUrl = URL.createObjectURL(blob);
        recordingState = "ready";
        stopMediaStream();
        mediaRecorder = null;
        recordingChunks = [];
      };
      mediaRecorder.start();
      recordingState = "recording";
    } catch (error) {
      stopMediaStream();
      mediaRecorder = null;
      recordingState = "idle";
      status = "error";
      errorMessage = error instanceof Error ? error.message : String(error);
    }
  }

  function stopRecording() {
    if (mediaRecorder?.state === "recording") {
      mediaRecorder.stop();
    }
  }

  async function run() {
    if (!artifactsReady) {
      status = "error";
      errorMessage = "Download the required artifacts before running FunASR";
      return;
    }
    if (!recordedAudio) {
      status = "error";
      errorMessage = "Record audio before running FunASR";
      return;
    }

    status = "running";
    errorMessage = null;
    transcript = "";
    generatedIds = [];
    runMeta = null;
    artifactDiagnostics = null;

    try {
      validateEncoderUrl(artifactUrls.encoder);
      const devices = await init("webgpu");
      if (!devices.includes("webgpu")) {
        throw new Error("WebGPU backend is not available in this browser");
      }
      defaultDevice("webgpu");

      const decodedWaveform = await decodeBrowserAudio(recordedAudio, frontendConfig.fs);
      const waveform = trimWaveformSilence(decodedWaveform);
      const extracted = await extractFunASRNanoSpeech(
        waveform,
        frontendConfig,
      );
      const { speech, shape } = normalizeSpeechFrames(
        extracted.speech,
        extracted.shape,
        encoderFrames,
      );

      artifactDiagnostics = {
        ...(artifactDiagnostics ?? {}),
        run_encoder_url: artifactUrls.encoder,
        run_encoder_cached: artifactCache.encoder?.cached ?? false,
        run_encoder_size_mb: Math.round((artifactCache.encoder?.size ?? 0) / 1024 / 1024),
      };
      const encoderBytes = await readArtifact("encoder");
      const encoder = new ONNXModel(encoderBytes);
      const encoderInputsDeclared = encoder.model.graph?.input.map((value) => value.name) ?? [];
      const encoderOutputsDeclared = encoder.model.graph?.output.map((value) => value.name) ?? [];
      const encoderOutputDimsDeclared = encoder.model.graph?.output.map((value) =>
        value.type?.value.case === "tensorType"
          ? value.type.value.value.shape?.dim.map((dim) => {
              if (dim.value.case === "dimValue") return Number(dim.value.value);
              if (dim.value.case === "dimParam") return String(dim.value.value);
              return null;
            })
          : null,
      ) ?? [];
      const encoderOps = [...new Set(encoder.model.graph?.node.map((node) => node.opType) ?? [])];
      artifactDiagnostics = {
        ...(artifactDiagnostics ?? {}),
        encoder_url: artifactUrls.encoder,
        encoder_cached: artifactCache.encoder?.cached ?? false,
        encoder_size_mb: Math.round((artifactCache.encoder?.size ?? encoderBytes.byteLength) / 1024 / 1024),
        encoder_inputs_declared: encoderInputsDeclared,
        encoder_outputs_declared: encoderOutputsDeclared,
        encoder_output_dims_declared: encoderOutputDimsDeclared,
        encoder_ops_declared: encoderOps,
      };
      if (encoderInputsDeclared.includes("input_lengths")) {
        throw new Error(
          "The cached encoder artifact is the raw encoder graph, not encoder_adaptor.onnx. Click Reset Cached Artifacts and download again.",
        );
      }
      const speechLengths = np.array(new Int32Array([encoderFrames]), {
        dtype: np.int32,
        shape: [1],
      });
      const {
        encoderOut,
        inputNames: encoderInputNames,
        outputNames: encoderOutputNames,
      } = runEncoderCompat(encoder, speech, shape, speechLengths);
      if (encoderOut.shape[encoderOut.shape.length - 1] !== 1024) {
        throw new Error(
          "The selected encoder artifact does not include the FunASR adaptor. Use an encoder_adaptor.onnx artifact with output width 1024.",
        );
      }
      artifactDiagnostics = {
        ...(artifactDiagnostics ?? {}),
        encoder_inputs: encoderInputNames,
        encoder_outputs: encoderOutputNames,
        encoder_output_shape: encoderOut.shape,
      };

      const tokenizer = tokenizers.HuggingFaceBPE.fromBinary(
        await readArtifact("tokenizer"),
      );
      const prepared = prepareFunASRNanoSourceIds(
        tokenizer,
        recordedAudio.name,
        shape[1],
      );

      const qwen = loadFunASRNanoQwenFromBuffers(
        await readArtifact("llm"),
        new TextDecoder().decode(await readArtifact("config")),
        new TextDecoder().decode(await readArtifact("generationConfig")),
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
        extracted_speech_shape: extracted.shape,
        decoded_waveform_length: decodedWaveform.length,
        waveform_length: waveform.length,
        audio_file: recordedAudio.name,
        input_device_label: inputDeviceLabel,
        input_device_id: inputDeviceId,
        output_device_label: outputDeviceLabel,
        encoder_inputs: encoderInputNames,
        encoder_outputs: encoderOutputNames,
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

  onMount(() => {
    refreshArtifactCache();
    refreshAudioInputDevices();
    refreshAudioOutputLabel();
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshAudioInputDevices);
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshAudioOutputLabel);
  });

  onDestroy(() => {
    navigator.mediaDevices?.removeEventListener?.("devicechange", refreshAudioInputDevices);
    navigator.mediaDevices?.removeEventListener?.("devicechange", refreshAudioOutputLabel);
  });
</script>

<Seo
  title="FunASR Nano WebGPU Runner"
  description="Browser-side WebGPU staging page for FunASR-nano in jax-js."
/>

<svelte:head>
  <title>FunASR Nano WebGPU Runner</title>
</svelte:head>

<DownloadManager bind:this={downloadManager} />

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
          files into the repo. Microphone audio is recorded and decoded directly
          in-browser before the
          frontend, encoder/adaptor, and Qwen decode path run on WebGPU.
        </p>

        <p class="mt-4 max-w-2xl text-sm leading-6 text-stone-500">
          Chrome microphone capture is the primary end-user path here. The model
          artifacts now download into browser-local cache and are only fetched
          when missing. Spoken input comes from direct browser recording.
        </p>

        <div class="mt-8 rounded-[1.75rem] border border-stone-300 bg-stone-50/90 p-5">
          <div class="mb-4 flex items-center justify-between gap-3">
            <div class="flex items-center gap-2 text-sm font-medium text-stone-700">
              <DownloadIcon size={16} />
              Required Artifacts
            </div>
            <button
              class="inline-flex items-center gap-2 rounded-full bg-stone-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-lime-800 disabled:cursor-not-allowed disabled:bg-stone-400"
              disabled={isPreparingArtifacts || status === "running"}
              onclick={prepareArtifacts}
            >
              <DownloadIcon size={16} />
              {isPreparingArtifacts ? "Preparing Cache…" : artifactsReady ? "Refresh Cache" : "Download Required Files"}
            </button>
            <button
              class="inline-flex items-center gap-2 rounded-full border border-stone-300 bg-white px-5 py-3 text-sm font-medium text-stone-700 transition hover:bg-stone-100 disabled:cursor-not-allowed disabled:bg-stone-100 disabled:text-stone-400"
              disabled={isPreparingArtifacts || status === "running"}
              onclick={resetArtifacts}
            >
              Reset Cached Artifacts
            </button>
          </div>

          <div class="grid gap-4 sm:grid-cols-2">
            {#each Object.entries(fileLabels) as [key, label]}
              <label class="rounded-2xl border border-stone-300 bg-white p-4">
                <div class="mb-2 flex items-center gap-2 text-sm font-medium text-stone-700">
                  <FileUpIcon size={16} />
                  {label}
                </div>
                <input
                  class="block w-full rounded-xl border border-stone-300 bg-stone-50 px-3 py-2 text-sm text-stone-700"
                  bind:value={artifactUrls[key as ArtifactKey]}
                  placeholder="Artifact URL"
                />
                {#if artifactCache[key as ArtifactKey]?.cached}
                  <p class="mt-2 text-xs text-lime-700">
                    Cached locally ({Math.round((artifactCache[key as ArtifactKey]?.size ?? 0) / 1024 / 1024)} MB)
                  </p>
                {:else}
                  <p class="mt-2 text-xs text-stone-500">Not cached yet.</p>
                {/if}
              </label>
            {/each}
          </div>
        </div>

        <div class="mt-8 rounded-[1.75rem] border border-stone-300 bg-stone-50/90 p-5">
          <div class="mb-3 flex items-center gap-2 text-sm font-medium text-stone-700">
            <MicIcon size={16} />
            Microphone Recording
          </div>
          <label class="mb-4 block">
            <span class="mb-2 block text-sm text-stone-600">Input device</span>
            <select
              class="block w-full rounded-xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-700 disabled:bg-stone-100"
              bind:value={selectedInputDeviceId}
              disabled={recordingState === "recording" || audioInputDevices.length === 0}
            >
              {#if audioInputDevices.length === 0}
                <option value="">No microphone devices detected yet</option>
              {:else}
                {#each audioInputDevices as device}
                  <option value={device.deviceId}>
                    {device.label || `Microphone ${device.deviceId.slice(0, 8)}`}
                  </option>
                {/each}
              {/if}
            </select>
          </label>
          <div class="flex flex-wrap items-center gap-3">
            <button
              class="inline-flex items-center gap-2 rounded-full bg-lime-700 px-5 py-3 text-sm font-medium text-white transition hover:bg-lime-800 disabled:cursor-not-allowed disabled:bg-stone-400"
              disabled={recordingState === "recording" || status === "running"}
              onclick={startRecording}
            >
              <MicIcon size={16} />
              {recordingState === "ready" ? "Record Again" : "Start Recording"}
            </button>
            <button
              class="inline-flex items-center gap-2 rounded-full bg-stone-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-400"
              disabled={recordingState !== "recording"}
              onclick={stopRecording}
            >
              <SquareIcon size={16} />
              Stop Recording
            </button>
            {#if recordingState === "recording"}
              <p class="text-sm text-rose-600">Recording in progress...</p>
            {:else if recordedAudio}
              <p class="text-sm text-stone-600">
                Recorded clip ready: {recordedAudio.name}
              </p>
            {:else}
              <p class="text-sm text-stone-500">
                Record a short utterance before running FunASR.
              </p>
            {/if}
          </div>
          <div class="mt-4 grid gap-2 text-sm text-stone-600">
            <p>
              Input device: {inputDeviceLabel || "No microphone selected yet"}
            </p>
            <p>
              Output device: {outputDeviceLabel || "System default audio output"}
            </p>
          </div>
          {#if recordedAudioUrl}
            <audio class="mt-4 w-full" controls src={recordedAudioUrl}></audio>
          {/if}
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
            disabled={status === "running" || !webgpuAvailable || recordingState === "recording" || !recordedAudio || !artifactsReady || isPreparingArtifacts}
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
              Download the model artifacts once, record a clip, and run the browser-side WebGPU path.
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

        <div class="rounded-[2rem] border border-stone-300/60 bg-white/90 p-6 shadow-[0_16px_50px_rgba(64,64,32,0.08)]">
          <p class="mb-3 text-sm uppercase tracking-[0.22em] text-stone-500">
            Artifact Diagnostics
          </p>
          <pre class="overflow-x-auto rounded-2xl bg-stone-100 p-4 text-sm leading-6 text-stone-700">{artifactDiagnostics ? JSON.stringify(artifactDiagnostics, null, 2) : "{}"}</pre>
        </div>
      </div>
    </div>
  </section>
</main>

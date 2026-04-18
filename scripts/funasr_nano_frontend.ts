import { numpy as np } from "../dist/index.js";

type FrontendConfig = {
  fs: number;
  window: string;
  n_mels: number;
  frame_length: number;
  frame_shift: number;
  lfr_m: number;
  lfr_n: number;
  upsacle_samples?: boolean;
};

function nextPowerOfTwo(n: number): number {
  let power = 1;
  while (power < n) power *= 2;
  return power;
}

function melScale(freq: number): number {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

function createWindow(windowSize: number, windowType: string): Float32Array {
  const win = new Float32Array(windowSize);
  if (windowType !== "hamming") {
    throw new Error(`Unsupported window type: ${windowType}`);
  }
  for (let i = 0; i < windowSize; i++) {
    win[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (windowSize - 1));
  }
  return win;
}

function createMelBanks(
  numBins: number,
  paddedWindowSize: number,
  sampleFreq: number,
  lowFreq: number = 20,
  highFreq: number = 0,
): Float32Array {
  const nyquist = 0.5 * sampleFreq;
  if (highFreq <= 0) highFreq += nyquist;

  const numFftBins = paddedWindowSize / 2;
  const fftBinWidth = sampleFreq / paddedWindowSize;
  const melLow = melScale(lowFreq);
  const melHigh = melScale(highFreq);
  const melDelta = (melHigh - melLow) / (numBins + 1);
  const bins = new Float32Array(numBins * (numFftBins + 1));

  for (let bin = 0; bin < numBins; bin++) {
    const leftMel = melLow + bin * melDelta;
    const centerMel = melLow + (bin + 1) * melDelta;
    const rightMel = melLow + (bin + 2) * melDelta;

    for (let fftBin = 0; fftBin < numFftBins; fftBin++) {
      const mel = melScale(fftBinWidth * fftBin);
      const upSlope = (mel - leftMel) / (centerMel - leftMel);
      const downSlope = (rightMel - mel) / (rightMel - centerMel);
      bins[bin * (numFftBins + 1) + fftBin] = Math.max(
        0,
        Math.min(upSlope, downSlope),
      );
    }
  }

  return bins;
}

function makeFrames(
  waveform: Float32Array,
  windowSize: number,
  windowShift: number,
): Float32Array[] {
  if (waveform.length < windowSize) return [];
  const frameCount = 1 + Math.floor((waveform.length - windowSize) / windowShift);
  const frames: Float32Array[] = new Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    const start = i * windowShift;
    frames[i] = waveform.slice(start, start + windowSize);
  }
  return frames;
}

function preprocessFrames(
  frames: Float32Array[],
  windowFn: Float32Array,
  paddedWindowSize: number,
): Float32Array {
  const windowSize = windowFn.length;
  const out = new Float32Array(frames.length * paddedWindowSize);
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];

    let mean = 0;
    for (let j = 0; j < windowSize; j++) mean += frame[j];
    mean /= windowSize;

    let prev = frame[0] - mean;
    out[i * paddedWindowSize] = (prev - 0.97 * prev) * windowFn[0];
    for (let j = 1; j < windowSize; j++) {
      const current = frame[j] - mean;
      out[i * paddedWindowSize + j] = (current - 0.97 * prev) * windowFn[j];
      prev = current;
    }
  }
  return out;
}

function applyLfr(
  inputs: Float32Array,
  frames: number,
  featDim: number,
  lfrM: number,
  lfrN: number,
): { data: Float32Array; frames: number; featDim: number } {
  const leftPad = Math.floor((lfrM - 1) / 2);
  const paddedFrames = frames + leftPad;
  const padded = new Float32Array(paddedFrames * featDim);

  for (let i = 0; i < leftPad; i++) {
    padded.set(inputs.subarray(0, featDim), i * featDim);
  }
  padded.set(inputs, leftPad * featDim);

  const outFrames = Math.ceil(frames / lfrN);
  const outFeatDim = featDim * lfrM;
  const out = new Float32Array(outFrames * outFeatDim);

  for (let i = 0; i < outFrames; i++) {
    const baseFrame = i * lfrN;
    for (let j = 0; j < lfrM; j++) {
      const srcFrame = Math.min(baseFrame + j, paddedFrames - 1);
      out.set(
        padded.subarray(srcFrame * featDim, (srcFrame + 1) * featDim),
        i * outFeatDim + j * featDim,
      );
    }
  }

  return { data: out, frames: outFrames, featDim: outFeatDim };
}

export async function extractFunASRNanoSpeech(
  waveform: Float32Array,
  config: FrontendConfig,
): Promise<{ speech: Float32Array; shape: [number, number, number] }> {
  const scaled =
    config.upsacle_samples === false
      ? waveform
      : Float32Array.from(waveform, (x) => x * (1 << 15));

  const windowSize = Math.floor(
    config.fs * (config.frame_length / 1000),
  );
  const windowShift = Math.floor(
    config.fs * (config.frame_shift / 1000),
  );
  const paddedWindowSize = nextPowerOfTwo(windowSize);
  const frames = makeFrames(scaled, windowSize, windowShift);
  const windowFn = createWindow(windowSize, config.window);
  const framed = preprocessFrames(frames, windowFn, paddedWindowSize);

  const real = np.array(framed, {
    shape: [frames.length, paddedWindowSize],
    dtype: np.float32,
  });
  const imag = np.zeros([frames.length, paddedWindowSize], { dtype: np.float32 });
  const fft = np.fft.fft({ real, imag }, -1);
  const spectrum = np
    .square(fft.real)
    .add(np.square(fft.imag))
    .slice([], [0, paddedWindowSize / 2 + 1]);

  const melBanks = createMelBanks(
    config.n_mels,
    paddedWindowSize,
    config.fs,
  );
  const melBank = np.array(melBanks, {
    shape: [config.n_mels, paddedWindowSize / 2 + 1],
    dtype: np.float32,
  });
  const mel = np.dot(spectrum, melBank.transpose());
  const logged = np.log(np.maximum(mel, Number.EPSILON));
  const loggedData = await logged.data();
  const lfr = applyLfr(
    loggedData,
    frames.length,
    config.n_mels,
    config.lfr_m,
    config.lfr_n,
  );

  return {
    speech: lfr.data,
    shape: [1, lfr.frames, lfr.featDim],
  };
}

import { cachedFetch } from "@jax-js/loaders";

export type LayoutRegion = {
  label: string;
  score: number;
  readOrder: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

const MODEL_URL =
  "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx";
const CONFIG_URL =
  "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/config.json";

const INPUT_SIZE = 800;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortSession: import("onnxruntime-web/webgpu").InferenceSession | null = null;
let idToLabel: Record<string, string> | null = null;

function isTextLikeLabel(label: string): boolean {
  const normalized = label.toLowerCase();
  if (
    normalized.includes("figure") ||
    normalized.includes("image") ||
    normalized.includes("picture") ||
    normalized.includes("barcode") ||
    normalized.includes("qr") ||
    normalized.includes("seal") ||
    normalized.includes("logo")
  ) return false;
  return true;
}

async function getOrt() {
  return await import("onnxruntime-web/webgpu");
}

async function getSession() {
  if (ortSession) return ortSession;
  const ort = await getOrt();
  const modelBytes = await cachedFetch(MODEL_URL);
  ortSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["webgpu"],
  });
  return ortSession;
}

async function getLabels() {
  if (idToLabel) return idToLabel;
  try {
    const data = await cachedFetch(CONFIG_URL);
    const json = JSON.parse(new TextDecoder().decode(data));
    idToLabel = json.id2label ?? {};
  } catch {
    idToLabel = {};
  }
  return idToLabel;
}

function sourceSize(source: HTMLImageElement | HTMLCanvasElement | ImageBitmap) {
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  return { width: source.width, height: source.height };
}

function preprocessLayoutImage(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
): { inputBlob: Float32Array; scaleH: number; scaleW: number } {
  const { width: origW, height: origH } = sourceSize(source);
  const scaleH = INPUT_SIZE / origH;
  const scaleW = INPUT_SIZE / origW;

  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(source as CanvasImageSource, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const blob = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < plane; i++) {
    blob[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    blob[plane + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    blob[2 * plane + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return { inputBlob: blob, scaleH, scaleW };
}

export function cropLayoutRegion(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  region: LayoutRegion,
): HTMLCanvasElement {
  const { width, height } = sourceSize(source);
  const x1 = Math.max(0, Math.floor(region.x1));
  const y1 = Math.max(0, Math.floor(region.y1));
  const x2 = Math.min(width, Math.ceil(region.x2));
  const y2 = Math.min(height, Math.ceil(region.y2));

  const crop = document.createElement("canvas");
  crop.width = Math.max(1, x2 - x1);
  crop.height = Math.max(1, y2 - y1);
  crop.getContext("2d")!.drawImage(
    source as CanvasImageSource,
    x1,
    y1,
    crop.width,
    crop.height,
    0,
    0,
    crop.width,
    crop.height,
  );
  return crop;
}

export async function analyzeLayout(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  threshold = 0.5,
): Promise<LayoutRegion[]> {
  const [session, labels, ort] = await Promise.all([getSession(), getLabels(), getOrt()]);
  const safeLabels = labels ?? {};
  const { inputBlob, scaleH, scaleW } = preprocessLayoutImage(source);

  const inputs: Record<string, import("onnxruntime-web/webgpu").Tensor> = {};
  const inputNames = session.inputNames;
  if (inputNames.length < 3) {
    throw new Error(`Unexpected PPDocLayoutV3 input signature: ${inputNames.join(", ")}`);
  }

  inputs[inputNames[0]] = new ort.Tensor("float32", new Float32Array([INPUT_SIZE, INPUT_SIZE]), [1, 2]);
  inputs[inputNames[1]] = new ort.Tensor("float32", inputBlob, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  inputs[inputNames[2]] = new ort.Tensor("float32", new Float32Array([scaleH, scaleW]), [1, 2]);

  const outputs = await session.run(inputs);
  const firstOutput = outputs[session.outputNames[0]];
  const data = firstOutput.data as Float32Array;
  const rows = data.length / 7;

  const regions: LayoutRegion[] = [];
  for (let i = 0; i < rows; i++) {
    const base = i * 7;
    const labelId = Math.round(data[base]);
    const score = data[base + 1];
    if (score < threshold) continue;

    const label = safeLabels[String(labelId)] ?? `class_${labelId}`;
    if (!isTextLikeLabel(label)) continue;

    const x1 = data[base + 2];
    const y1 = data[base + 3];
    const x2 = data[base + 4];
    const y2 = data[base + 5];
    const readOrder = Math.round(data[base + 6]);
    if (x2 <= x1 || y2 <= y1) continue;

    regions.push({ label, score, readOrder, x1, y1, x2, y2 });
  }

  regions.sort((a, b) => a.readOrder - b.readOrder || a.y1 - b.y1 || a.x1 - b.x1);
  return regions;
}

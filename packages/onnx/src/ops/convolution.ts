// Convolution operations.

import { lax, numpy as np } from "@jax-js/jax";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
} from "../tensor";

const padsMapping: Record<string, lax.PaddingType> = {
  SAME_UPPER: "SAME",
  SAME_LOWER: "SAME_LOWER",
  VALID: "VALID",
};

export function Conv(
  inputs: Operand[],
  {
    auto_pad: autoPad = "NOTSET",
    dilations,
    group = 1,
    kernel_shape: _kernelShape, // inferred from weights
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    dilations?: number[];
    group?: number;
    kernel_shape?: number[];
    pads?: number[];
    strides?: number[];
  },
): Operand[] {
  const [x, w, bias] = inputs.map(operandToJax);
  if (!x || !w) throw new Error("Conv: missing required inputs");
  const [_batchSize, channelsIn, ...xSpatial] = x.shape;
  const [_channelsOut, channelsInGrouped, ...wSpatial] = w.shape;
  if (channelsIn !== channelsInGrouped * group) {
    throw new Error(
      `Conv: input channels ${channelsIn} must match weight channels ${channelsInGrouped} x group ${group}`,
    );
  }
  if (xSpatial.length !== wSpatial.length) {
    throw new Error(
      `Conv: input spatial dims ${xSpatial.length} must match weight spatial dims ${wSpatial.length}`,
    );
  }
  const n = xSpatial.length;
  let output = lax.convGeneralDilated(
    x,
    w,
    strides ?? wSpatial.map(() => 1),
    padsMapping[autoPad] ??
      pads?.slice(0, n).map((p, i) => [p, pads[i + n]]) ??
      "VALID",
    {
      rhsDilation: dilations,
      featureGroupCount: group,
    },
  );
  // Add bias if provided (reshape to [1, C, 1, 1, ...] for broadcasting)
  if (bias) {
    const biasShape = [bias.size, ...xSpatial.map(() => 1)];
    output = output.add(bias.reshape(biasShape));
  }
  return [output];
}

// Pad a tensor with -Infinity along spatial dimensions (for max pooling).
function padWithNegInf(x: np.Array, pads: [number, number][]): np.Array {
  // pads is for spatial dims only, we need to add batch and channel dims
  for (let i = 0; i < pads.length; i++) {
    const [padBefore, padAfter] = pads[i];
    const axis = i + 2; // Skip batch and channel dims
    if (padBefore > 0) {
      const beforeShape = [...x.shape];
      beforeShape[axis] = padBefore;
      const before = np.full(beforeShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([before, x], axis);
    }
    if (padAfter > 0) {
      const afterShape = [...x.shape];
      afterShape[axis] = padAfter;
      const after = np.full(afterShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([x, after], axis);
    }
  }
  return x;
}

export function MaxPool(
  [xOp]: Operand[],
  {
    auto_pad: autoPad = "NOTSET",
    ceil_mode: ceilMode = 0,
    dilations,
    kernel_shape: kernelShape,
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    ceil_mode?: number;
    dilations?: number[];
    kernel_shape: number[];
    pads?: number[];
    strides?: number[];
  },
): Operand[] {
  if (ceilMode) {
    throw new Error("MaxPool: ceil_mode=1 is not supported");
  }
  if (dilations && dilations.some((d) => d !== 1)) {
    throw new Error("MaxPool: dilations != 1 is not supported");
  }
  const x = operandToJax(xOp);
  const n = kernelShape.length;
  const xSpatial = x.shape.slice(2);
  if (xSpatial.length !== n) {
    throw new Error(
      `MaxPool: input spatial dims ${xSpatial.length} must match kernel dims ${n}`,
    );
  }

  // Compute explicit padding
  let explicitPads: [number, number][];
  if (autoPad !== "NOTSET") {
    const effectiveStrides = strides ?? kernelShape.map(() => 1);
    const outShape = xSpatial.map((size, i) =>
      Math.ceil(size / effectiveStrides[i]),
    );
    const padSizes = outShape.map((o, i) => {
      const s = effectiveStrides[i];
      const k = kernelShape[i];
      const inSize = xSpatial[i];
      return Math.max(0, (o - 1) * s + k - inSize);
    });
    explicitPads =
      autoPad === "SAME_UPPER"
        ? padSizes.map((size) => [size >> 1, size - (size >> 1)])
        : padSizes.map((size) => [size - (size >> 1), size >> 1]);
  } else if (pads) {
    explicitPads = pads
      .slice(0, n)
      .map((p, i) => [p, pads[i + n]] as [number, number]);
  } else {
    explicitPads = kernelShape.map(() => [0, 0] as [number, number]);
  }

  // Apply padding with -Infinity if needed
  const needsPadding = explicitPads.some(([a, b]) => a > 0 || b > 0);
  const padded = needsPadding ? padWithNegInf(x, explicitPads) : x;

  const output = lax.reduceWindow(
    padded,
    np.max,
    kernelShape,
    strides ?? kernelShape.map(() => 1),
  );
  return [output];
}

export function Resize(
  [xOp, roi, scales, sizes]: Operand[],
  {
    coordinate_transformation_mode: coordMode = "half_pixel",
    mode = "nearest",
    nearest_mode: nearestMode = "round_prefer_floor",
  }: {
    coordinate_transformation_mode?: string;
    mode?: string;
    nearest_mode?: string;
    // Ignored: cubic_coeff_a, exclude_outside, extrapolation_value,
    // keep_aspect_ratio_policy, axes
  },
): Operand[] {
  // Only support nearest + asymmetric + floor for now
  if (mode !== "nearest") {
    throw new Error(`Resize: mode '${mode}' is not supported, only 'nearest'`);
  }
  if (coordMode !== "asymmetric") {
    throw new Error(
      `Resize: coordinate_transformation_mode '${coordMode}' is not supported, only 'asymmetric'`,
    );
  }
  if (nearestMode !== "floor") {
    throw new Error(
      `Resize: nearest_mode '${nearestMode}' is not supported, only 'floor'`,
    );
  }

  if (roi && !(roi instanceof StaticArray)) {
    // We don't use roi, so just dispose it.
    roi.dispose();
  }

  // Determine output shape from scales or sizes
  const x = operandToJax(xOp);
  const inShape = x.shape;
  let outShape: number[];
  if (sizes && sizes.shape[0] > 0) {
    outShape = operandToJs(sizes);
  } else if (scales && scales.shape[0] > 0) {
    const scalesArr: number[] = operandToJs(scales);
    outShape = inShape.map((d, i) => Math.floor(d * scalesArr[i]));
  } else {
    throw new Error("Resize: either scales or sizes must be provided");
  }

  // For asymmetric + nearest + floor:
  // input_coord = floor(output_coord * (input_size / output_size))
  // This is equivalent to: input_coord = floor(output_coord / scale)
  //
  // We implement this by creating index arrays for each dimension and using
  // advanced indexing.
  let result = x;
  for (let axis = 0; axis < inShape.length; axis++) {
    const inSize = result.shape[axis];
    const outSize = outShape[axis];
    if (inSize === outSize) continue;

    // Create indices: floor(i * inSize / outSize) for i in 0..outSize
    const indices = np.array(
      Array.from({ length: outSize }, (_, i) =>
        Math.floor((i * inSize) / outSize),
      ),
      { dtype: np.int32 },
    );

    // Build slice args: [] for all dims except current axis
    const sliceArgs: (np.Array | [])[] = result.shape.map(() => [] as []);
    sliceArgs[axis] = indices;
    result = result.slice(...sliceArgs);
  }

  return [result];
}

/**
 * ConvTranspose (ONNX opset 11).
 *
 * Transposed convolution (deconvolution).
 * W shape: [C_in, C_out/group, *kernel]
 */
export function ConvTranspose(
  inputs: Operand[],
  {
    auto_pad: autoPad = "NOTSET",
    dilations,
    group = 1,
    kernel_shape: _kernelShape,
    output_padding: outputPadding,
    output_shape: _outputShape,
    pads,
    strides,
  }: {
    auto_pad?: string;
    dilations?: number[];
    group?: number;
    kernel_shape?: number[];
    output_padding?: number[];
    output_shape?: number[];
    pads?: number[];
    strides?: number[];
  },
): Operand[] {
  const [xOp2, wOp2, biasOp2] = inputs;
  const x = operandToJax(xOp2);
  const w = operandToJax(wOp2);
  const bias = biasOp2 ? operandToJax(biasOp2) : null;

  const n = x.ndim - 2; // number of spatial dims
  const s = strides ?? new Array(n).fill(1);
  const d = dilations ?? new Array(n).fill(1);

  let padding: lax.PaddingType;
  if (autoPad !== "NOTSET" && padsMapping[autoPad]) {
    padding = padsMapping[autoPad];
  } else if (pads) {
    padding = pads.slice(0, n).map((p, i) => [p, pads[i + n]] as [number, number]);
  } else {
    padding = "VALID";
  }

  // W is [C_in, C_out/group, *k] in ONNX → use transposeKernel=true
  let output = lax.convTranspose(x, w, s, padding, {
    rhsDilation: d,
    transposeKernel: true,
  });

  // Add output_padding (extra zeros on the right/bottom)
  if (outputPadding && outputPadding.some((p) => p > 0)) {
    for (let i = 0; i < n; i++) {
      if (outputPadding[i] > 0) {
        const padShape = [...output.shape];
        padShape[i + 2] = outputPadding[i];
        const pad = np.zeros(padShape, { dtype: output.dtype });
        output = np.concatenate([output, pad], i + 2);
      }
    }
  }

  if (bias) {
    const biasShape = [1, bias.size, ...new Array(n).fill(1)];
    output = output.add(bias.reshape(biasShape));
  }

  return [output];
}

/**
 * RoiAlign (ONNX opset 16).
 *
 * Region of Interest align pooling with bilinear interpolation.
 * Executed on CPU (materializes tensors).
 *
 * X:            [N, C, H, W]
 * rois:         [num_rois, 4] (x1, y1, x2, y2) or [num_rois, 5] (n, x1, y1, x2, y2)
 * batch_indices: [num_rois] (optional, used when rois has 4 columns)
 */
export function RoiAlign(
  [xOp, roisOp, batchIndicesOp]: Operand[],
  {
    mode = "avg",
    output_height: outputH = 1,
    output_width: outputW = 1,
    sampling_ratio: samplingRatio = 0,
    spatial_scale: spatialScale = 1.0,
    coordinate_transformation_mode: coordMode = "half_pixel",
  }: {
    mode?: string;
    output_height?: number;
    output_width?: number;
    sampling_ratio?: number;
    spatial_scale?: number;
    coordinate_transformation_mode?: string;
  },
): Operand[] {
  const xData: number[] = operandToJs(xOp);
  const roisData: number[] = operandToJs(roisOp);
  const [, C, H, W] = xOp.shape;
  const roisShape = roisOp.shape;
  const numRois = roisShape[0];
  const roisHas5 = roisShape[1] === 5;

  // Batch indices
  let batchIdx: number[];
  if (roisHas5) {
    batchIdx = Array.from({ length: numRois }, (_, i) => roisData[i * 5]);
  } else if (batchIndicesOp) {
    batchIdx = operandToJs(batchIndicesOp);
  } else {
    batchIdx = new Array(numRois).fill(0);
  }

  const roiOffset = roisHas5 ? 1 : 0;

  const output = new Float32Array(numRois * C * outputH * outputW);

  for (let r = 0; r < numRois; r++) {
    const n = batchIdx[r];
    const x1 = roisData[r * roisShape[1] + roiOffset + 0] * spatialScale;
    const y1 = roisData[r * roisShape[1] + roiOffset + 1] * spatialScale;
    const x2 = roisData[r * roisShape[1] + roiOffset + 2] * spatialScale;
    const y2 = roisData[r * roisShape[1] + roiOffset + 3] * spatialScale;

    const roiW = Math.max(x2 - x1, 1);
    const roiH = Math.max(y2 - y1, 1);
    const binW = roiW / outputW;
    const binH = roiH / outputH;
    const sr = samplingRatio > 0 ? samplingRatio : Math.max(1, Math.ceil(Math.max(binH, binW)));

    for (let c = 0; c < C; c++) {
      const featureBase = (n * C + c) * H * W;
      for (let ph = 0; ph < outputH; ph++) {
        for (let pw = 0; pw < outputW; pw++) {
          let sum = 0;
          let count = 0;
          for (let iy = 0; iy < sr; iy++) {
            for (let ix = 0; ix < sr; ix++) {
              // Sample point in feature map coordinates
              let fy = y1 + binH * (ph + (iy + 0.5) / sr);
              let fx = x1 + binW * (pw + (ix + 0.5) / sr);

              if (coordMode === "output_half_pixel") {
                fy -= 0.5;
                fx -= 0.5;
              }

              if (fy < -1 || fy > H || fx < -1 || fx > W) {
                count++;
                continue;
              }
              fy = Math.max(0, Math.min(fy, H - 1));
              fx = Math.max(0, Math.min(fx, W - 1));

              const yLow = Math.floor(fy);
              const xLow = Math.floor(fx);
              const yHigh = Math.min(yLow + 1, H - 1);
              const xHigh = Math.min(xLow + 1, W - 1);

              const ly = fy - yLow;
              const lx = fx - xLow;
              const hy = 1 - ly;
              const hx = 1 - lx;

              // Bilinear interpolation
              sum +=
                hy * hx * xData[featureBase + yLow * W + xLow] +
                hy * lx * xData[featureBase + yLow * W + xHigh] +
                ly * hx * xData[featureBase + yHigh * W + xLow] +
                ly * lx * xData[featureBase + yHigh * W + xHigh];
              count++;
            }
          }
          const outIdx = ((r * C + c) * outputH + ph) * outputW + pw;
          output[outIdx] = count > 0 ? sum / count : 0;
        }
      }
    }
  }

  return [np.array(output, { shape: [numRois, C, outputH, outputW] })];
}

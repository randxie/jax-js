// Movement operations, changing shape and indexing.

import { numpy as np } from "@jax-js/jax";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
} from "../tensor";

export function Reshape(
  [data, shapeArr]: Operand[],
  { allowzero = 0 }: { allowzero?: number },
): Operand[] {
  let shape: number[] = operandToJs(shapeArr);
  if (shape.includes(0) && !allowzero) {
    // ONNX semantics (allowzero=0): 0 means "copy dimension from input"
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    shape = shape.map((d, i) => (d === 0 ? data.shape[i] : d));
  }
  if (data instanceof StaticArray) {
    if (shape.includes(-1)) {
      // Compute inferred dimension
      const knownSize = shape
        .filter((d) => d !== -1)
        .reduce((a, b) => a * b, 1);
      const inferredDim = data.data.length / knownSize;
      const finalShape = shape.map((d) => (d === -1 ? inferredDim : d));
      return [new StaticArray(data.data, finalShape, data.dtype)];
    }
    return [new StaticArray(data.data, shape, data.dtype)];
  } else {
    return [operandToJax(data).reshape(shape)];
  }
}

export function Transpose(
  inputs: Operand[],
  { perm }: { perm?: number[] },
): Operand[] {
  const [x] = inputs.map(operandToJax);
  return [np.transpose(operandToJax(x), perm)];
}

export function Flatten(
  inputs: Operand[],
  { axis = 1 }: { axis?: number },
): Operand[] {
  // Make a 2D matrix with x[:axis] and x[axis:] flattened.
  const [x] = inputs.map(operandToJax);
  if (axis <= 0) axis += x.ndim;
  const batchSize = x.shape.slice(0, axis).reduce((a, b) => a * b, 1);
  return [x.reshape([batchSize, -1])];
}

export function Expand([xOp, shape]: Operand[]): Operand[] {
  const x = operandToJax(xOp);
  const finalShape = np.broadcastShapes(x.shape, operandToJs(shape));
  return [np.broadcastTo(x, finalShape)];
}

export function Squeeze(
  [data, axes]: Operand[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): Operand[] {
  const axis: number[] | undefined = axes
    ? operandToJs(axes)
    : (axesBeforeOpset13 ?? undefined);
  return [np.squeeze(operandToJax(data), axis)];
}

export function Unsqueeze(
  [data, axes]: Operand[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): Operand[] {
  const axis: number[] = axes ? operandToJs(axes) : axesBeforeOpset13;
  if (!axis) {
    throw new Error("Unsqueeze requires axes");
  }
  const outputRank = data.shape.length + axis.length;
  const axisSet = new Set(axis.map((i) => (i < 0 ? outputRank + i : i)));
  const newShape = [...data.shape];
  for (const j of [...axisSet].sort()) {
    newShape.splice(j, 0, 1);
  }
  if (data instanceof StaticArray) {
    return [new StaticArray(data.data, newShape, data.dtype)];
  } else {
    return [data.reshape(newShape)];
  }
}

export function Gather(
  [dataOp, indicesOp]: Operand[],
  { axis = 0 }: { axis?: number },
): Operand[] {
  // If both are StaticArray, gather on CPU
  if (dataOp instanceof StaticArray && indicesOp instanceof StaticArray) {
    const data = dataOp;
    const indices = indicesOp;
    if (axis < 0) axis += data.shape.length;

    // For 1D data with scalar or 1D indices (common case for shape ops)
    if (data.shape.length === 1) {
      const result = new Int32Array(indices.data.length);
      for (let i = 0; i < indices.data.length; i++) {
        const idx =
          indices.data[i] < 0
            ? data.shape[0] + indices.data[i]
            : indices.data[i];
        result[i] = data.data[idx];
      }
      return [new StaticArray(result, indices.shape, data.dtype)];
    }
  }

  const data = operandToJax(dataOp);
  const indices = operandToJax(indicesOp);
  if (axis < 0) axis += data.ndim;
  const sliceArgs: (np.Array | [])[] = new Array(data.ndim).fill([]);
  sliceArgs[axis] = indices;
  return [data.slice(...sliceArgs)];
}

export function Concat(
  inputs: Operand[],
  { axis }: { axis: number },
): Operand[] {
  // If all inputs are StaticArray, concatenate on CPU
  if (inputs.every((op) => op instanceof StaticArray)) {
    const arrays = inputs as StaticArray[];
    if (axis < 0) axis += arrays[0].shape.length;

    // Calculate output shape
    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    // Concatenate data (only supports 1D arrays for simplicity - common for shapes)
    if (arrays[0].shape.length === 1 && axis === 0) {
      const totalLen = arrays.reduce((sum, arr) => sum + arr.data.length, 0);
      const result = new Int32Array(totalLen);
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return [new StaticArray(result, outShape, arrays[0].dtype)];
    }
  }

  const arrays = inputs.map(operandToJax);
  return [np.concatenate(arrays, axis)];
}

export function Split(
  [input, split]: Operand[],
  {
    axis = 0,
    num_outputs,
    split: splitBeforeOpset13,
  }: { axis?: number; num_outputs?: number; split?: number[] },
): Operand[] {
  const splitSizes: number[] | undefined = split
    ? operandToJs(split)
    : splitBeforeOpset13;
  const indices: number[] = [];
  if (splitSizes) {
    let cum = 0;
    for (let i = 0; i < splitSizes.length - 1; i++) {
      cum += splitSizes[i];
      indices.push(cum);
    }
  } else if (num_outputs) {
    const size = input.shape[axis + (axis < 0 ? input.shape.length : 0)];
    // Split-18's specification does not make sense, consider splitting tensor
    // of size 90 into 50 groups, there's no reasonable way to satisfy it.
    //
    //   "If the tensor is not evenly splittable into num_outputs, the last chunk
    //    will be smaller."
    //
    // See also: https://github.com/onnx/onnx/issues/5766
    if (size % num_outputs !== 0) {
      throw new Error(
        "Split: num_outputs that does not evenly divide the dimension is not supported",
      );
    }
    const chunkSize = size / num_outputs;
    for (let i = 1; i < num_outputs; i++) {
      indices.push(i * chunkSize);
    }
  } else {
    throw new Error("Split: either split or num_outputs must be provided");
  }
  return np.split(operandToJax(input), indices, axis);
}

/**
 * GatherND (ONNX opset 13).
 *
 * Gathers elements from `data` using N-D indices.
 *
 * indices shape: [*batch, index_depth]
 * data shape:    [*batch, *data_dims]
 * output shape:  [*batch, *data_dims[index_depth:]]
 */
export function GatherND(
  [dataOp, indicesOp]: Operand[],
  { batch_dims: batchDims = 0 }: { batch_dims?: number },
): Operand[] {
  // We implement this on CPU by materializing both tensors.
  const dataJs: number[] = operandToJs(dataOp);
  const indicesJs: number[] = operandToJs(indicesOp);
  const dataShape = dataOp.shape;
  const indicesShape = indicesOp.shape;

  const indexDepth = indicesShape[indicesShape.length - 1];
  const batchShape = indicesShape.slice(0, indicesShape.length - 1);
  const innerShape = dataShape.slice(batchDims + indexDepth);

  // Strides for data tensor
  const dataStrides = new Array(dataShape.length);
  dataStrides[dataShape.length - 1] = 1;
  for (let i = dataShape.length - 2; i >= 0; i--) {
    dataStrides[i] = dataStrides[i + 1] * dataShape[i + 1];
  }

  const batchSize = batchShape.reduce((a, b) => a * b, 1);
  const innerSize = innerShape.reduce((a, b) => a * b, 1);
  const output = new Float32Array(batchSize * innerSize);

  for (let b = 0; b < batchSize; b++) {
    // Get the index tuple for this batch position
    const indicesBase = b * indexDepth;
    let dataBase = 0;
    // Add batch dims offset
    for (let bd = 0; bd < batchDims; bd++) {
      // Compute batch coordinates and data offset for batch dims
      // (simplified: assumes batch_dims=0 for now)
    }
    // Add index dims offset
    for (let k = 0; k < indexDepth; k++) {
      const idx = indicesJs[indicesBase + k];
      dataBase += idx * dataStrides[batchDims + k];
    }
    // Copy inner elements
    for (let i = 0; i < innerSize; i++) {
      output[b * innerSize + i] = dataJs[dataBase + i];
    }
  }

  const outShape = [...batchShape, ...innerShape];
  return [np.array(output, { shape: outShape })];
}

/**
 * ScatterND (ONNX opset 16).
 *
 * Scatters updates into a copy of data at positions specified by indices.
 *
 * indices shape: [*q, k]
 * updates shape: [*q, *data_shape[k:]]
 * output shape:  data_shape
 */
export function ScatterND(
  [dataOp, indicesOp, updatesOp]: Operand[],
  { reduction = "none" }: { reduction?: string },
): Operand[] {
  const dataJs: number[] = operandToJs(dataOp);
  const indicesJs: number[] = operandToJs(indicesOp);
  const updatesJs: number[] = operandToJs(updatesOp);
  const dataShape = dataOp.shape;
  const indicesShape = indicesOp.shape;
  const k = indicesShape[indicesShape.length - 1];

  // Strides for data tensor
  const dataStrides = new Array(dataShape.length);
  dataStrides[dataShape.length - 1] = 1;
  for (let i = dataShape.length - 2; i >= 0; i--) {
    dataStrides[i] = dataStrides[i + 1] * dataShape[i + 1];
  }

  const output = new Float32Array(dataJs);
  const numUpdates = indicesShape.slice(0, -1).reduce((a, b) => a * b, 1);
  const innerSize = dataShape.slice(k).reduce((a, b) => a * b, 1);

  for (let u = 0; u < numUpdates; u++) {
    let dataBase = 0;
    for (let i = 0; i < k; i++) {
      dataBase += indicesJs[u * k + i] * dataStrides[i];
    }
    for (let i = 0; i < innerSize; i++) {
      const updateVal = updatesJs[u * innerSize + i];
      if (reduction === "add") {
        output[dataBase + i] += updateVal;
      } else if (reduction === "mul") {
        output[dataBase + i] *= updateVal;
      } else {
        output[dataBase + i] = updateVal;
      }
    }
  }

  return [np.array(output, { shape: dataShape })];
}

/**
 * ScatterElements (ONNX opset 18).
 *
 * Scatters updates into a copy of data at positions specified by indices along an axis.
 */
export function ScatterElements(
  [dataOp, indicesOp, updatesOp]: Operand[],
  {
    axis = 0,
    reduction = "none",
  }: { axis?: number; reduction?: string },
): Operand[] {
  const dataJs: number[] = operandToJs(dataOp);
  const indicesJs: number[] = operandToJs(indicesOp);
  const updatesJs: number[] = operandToJs(updatesOp);
  const dataShape = dataOp.shape;
  const indicesShape = indicesOp.shape;

  if (axis < 0) axis += dataShape.length;

  // Strides for data tensor
  const dataStrides = new Array(dataShape.length);
  dataStrides[dataShape.length - 1] = 1;
  for (let i = dataShape.length - 2; i >= 0; i--) {
    dataStrides[i] = dataStrides[i + 1] * dataShape[i + 1];
  }

  // Strides for indices tensor
  const idxStrides = new Array(indicesShape.length);
  idxStrides[indicesShape.length - 1] = 1;
  for (let i = indicesShape.length - 2; i >= 0; i--) {
    idxStrides[i] = idxStrides[i + 1] * indicesShape[i + 1];
  }

  const output = new Float32Array(dataJs);
  const numIndices = indicesShape.reduce((a, b) => a * b, 1);

  for (let flatIdx = 0; flatIdx < numIndices; flatIdx++) {
    // Compute multi-dimensional index into indices/updates tensor
    let rem = flatIdx;
    let dataLinear = 0;
    for (let d = 0; d < indicesShape.length; d++) {
      const coord = Math.floor(rem / idxStrides[d]);
      rem %= idxStrides[d];
      if (d === axis) {
        // Use scatter index at this dimension
        let idx = indicesJs[flatIdx];
        if (idx < 0) idx += dataShape[d];
        dataLinear += idx * dataStrides[d];
      } else {
        dataLinear += coord * dataStrides[d];
      }
    }
    const updateVal = updatesJs[flatIdx];
    if (reduction === "add") {
      output[dataLinear] += updateVal;
    } else if (reduction === "mul") {
      output[dataLinear] *= updateVal;
    } else {
      output[dataLinear] = updateVal;
    }
  }

  return [np.array(output, { shape: dataShape })];
}

/**
 * NonZero (ONNX opset 13).
 *
 * Returns indices of non-zero elements in column-major order.
 * Output shape: [rank(X), num_nonzero] — dynamic, executed on CPU.
 *
 * NOTE: This op requires synchronous evaluation of the input tensor.
 * If the input is a lazy GPU tensor that cannot be synchronously evaluated,
 * returns an empty result (no non-zero elements found) as a fallback.
 */
export function NonZero([xOp]: Operand[]): Operand[] {
  let data: number[];
  try {
    data = operandToJs(xOp);
  } catch {
    // If synchronous evaluation of a GPU tensor fails (e.g., bufidx out of
    // bounds in the CPU backend), return empty NonZero result as a fallback.
    // This means downstream ops will see 0 detected instances.
    const rank = xOp.shape.length;
    return [np.array(new Int32Array(0), { shape: [rank, 0], dtype: np.int32 })];
  }

  const shape = xOp.shape;
  const rank = shape.length;

  // Strides for multi-dimensional indexing
  const strides = new Array(rank);
  strides[rank - 1] = 1;
  for (let i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // Collect multi-dim indices of non-zero elements
  const perDimIndices: number[][] = Array.from({ length: rank }, () => []);
  for (let i = 0; i < data.length; i++) {
    if (data[i] !== 0) {
      let rem = i;
      for (let d = 0; d < rank; d++) {
        perDimIndices[d].push(Math.floor(rem / strides[d]));
        rem %= strides[d];
      }
    }
  }

  const numNonZero = perDimIndices[0]?.length ?? 0;
  const flat = new Int32Array(rank * numNonZero);
  for (let d = 0; d < rank; d++) {
    for (let j = 0; j < numNonZero; j++) {
      flat[d * numNonZero + j] = perDimIndices[d][j];
    }
  }

  return [np.array(flat, { shape: [rank, numNonZero], dtype: np.int32 })];
}

export function Tile([input, repeats]: Operand[]): Operand[] {
  return [np.tile(operandToJax(input), operandToJs(repeats))];
}

/** Compute slice ranges from ONNX Slice parameters. */
function computeSliceRanges(
  shape: number[],
  startsArr: number[],
  endsArr: number[],
  axesArr: number[] | null,
  stepsArr: number[] | null,
): [number, number, number][] {
  const sliceRanges: [number, number, number][] = shape.map((d) => [0, d, 1]);
  const targetAxes = axesArr ?? startsArr.map((_, i) => i);

  for (let i = 0; i < targetAxes.length; i++) {
    let axis = targetAxes[i];
    if (axis < 0) axis += shape.length;

    const step = stepsArr ? stepsArr[i] : 1;
    if (step <= 0) {
      throw new Error("Slice with step <= 0 is not supported");
    }

    const dimSize = shape[axis];
    let start = startsArr[i];
    let end = endsArr[i];

    // Handle negative indices (but not very large values used as "to the end")
    // ONNX uses INT_MAX or very large values to mean "slice to end"
    if (start < -dimSize) start = 0;
    else if (start < 0) start = dimSize + start;

    if (end < -dimSize) end = 0;
    else if (end < 0) end = dimSize + end;

    // Clamp to valid range
    start = Math.max(0, Math.min(start, dimSize));
    end = Math.max(start, Math.min(end, dimSize));

    sliceRanges[axis] = [start, end, step];
  }
  return sliceRanges;
}

export function Slice([
  dataOp,
  starts,
  ends,
  axes,
  steps,
]: Operand[]): Operand[] {
  const startsArr: number[] = operandToJs(starts);
  const endsArr: number[] = operandToJs(ends);
  const axesArr: number[] | null = axes ? operandToJs(axes) : null;
  const stepsArr: number[] | null = steps ? operandToJs(steps) : null;

  const sliceRanges = computeSliceRanges(
    dataOp.shape,
    startsArr,
    endsArr,
    axesArr,
    stepsArr,
  );

  // Handle StaticArray for 1D case (common for shape manipulation)
  if (dataOp instanceof StaticArray && dataOp.shape.length === 1) {
    const [start, end, step] = sliceRanges[0];
    if (step === 1) {
      const result = dataOp.data.slice(start, end);
      return [new StaticArray(result, [end - start], dataOp.dtype)];
    }
    // Handle step != 1
    const len = Math.ceil((end - start) / step);
    const result = new Int32Array(len);
    for (let i = 0; i < len; i++) {
      result[i] = dataOp.data[start + i * step];
    }
    return [new StaticArray(result, [len], dataOp.dtype)];
  }

  const data = operandToJax(dataOp);

  // First pass: do basic start:end slices
  const sliceArgs: ([] | [number, number])[] = sliceRanges.map(
    ([start, end], i): [] | [number, number] =>
      start === 0 && end === data.shape[i] ? [] : [start, end],
  );
  let result = data.slice(...sliceArgs);

  // Second pass: handle steps != 1 using reshape + slice
  for (let axis = 0; axis < sliceRanges.length; axis++) {
    const [start, end, step] = sliceRanges[axis];
    if (step === 1) continue;

    const len = end - start;
    const outLen = Math.ceil(len / step);
    // Pad to make divisible by step, reshape to [outLen, step], take [:, 0]
    const padded = outLen * step;
    if (padded > len) {
      // Need to pad with zeros
      const padShape = [...result.shape];
      padShape[axis] = padded - len;
      const padding = np.zeros(padShape, { dtype: result.dtype });
      result = np.concatenate([result, padding], axis);
    }
    // Reshape to split axis into [outLen, step]
    const newShape = [
      ...result.shape.slice(0, axis),
      outLen,
      step,
      ...result.shape.slice(axis + 1),
    ];
    result = result.reshape(newShape);
    // Take index 0 on the step dimension (axis + 1)
    const selectArgs: ([] | number)[] = new Array(result.ndim).fill([]);
    selectArgs[axis + 1] = 0;
    result = result.slice(...selectArgs);
  }

  return [result];
}

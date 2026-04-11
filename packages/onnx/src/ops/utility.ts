// Utility operations, such as dtype conversion and data prep.
//
// TODO: Range (prompt_encoder_mask_decoder, vision_encoder)
// TODO: OneHot (prompt_encoder_mask_decoder)

import { numpy as np } from "@jax-js/jax";
import { TensorProto } from "onnx-buf";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
  tensorToOperand,
} from "../tensor";

export function Shape(
  [data]: Operand[],
  { start = 0, end }: { start?: number; end?: number },
): Operand[] {
  const arr = operandToJax(data);
  const shape = arr.shape.slice(start, end);
  arr.dispose();
  return [new StaticArray(shape, [shape.length], np.int32)];
}

export function Constant(
  _: Operand[],
  {
    value,
    value_float,
    value_floats,
    value_int,
    value_ints,
    value_string,
    value_strings,
  }: {
    value?: TensorProto;
    value_float?: number;
    value_floats?: number[];
    value_int?: number;
    value_ints?: number[];
    value_string?: Uint8Array<ArrayBuffer>;
    value_strings?: Uint8Array<ArrayBuffer>[];
  },
): Operand[] {
  if (value !== undefined) {
    return [tensorToOperand(value)];
  } else if (value_float !== undefined) {
    return [np.array(value_float)];
  } else if (value_floats !== undefined) {
    return [np.array(value_floats)];
  } else if (value_int !== undefined) {
    return [new StaticArray([value_int], [], np.int32)];
  } else if (value_ints !== undefined) {
    return [new StaticArray(value_ints, [value_ints.length], np.int32)];
  } else if (value_string !== undefined || value_strings !== undefined) {
    throw new Error("ONNX Constant string values are not supported");
  } else {
    throw new Error("ONNX Constant has no value");
  }
}

/**
 * LayerNormalization (ONNX opset 17).
 *
 * Normalizes the input tensor along the axes [axis, ..., rank-1], then
 * scales and shifts using learnable parameters.
 *
 * Y = (X - mean) / sqrt(var + epsilon) * scale + bias
 */
export function LayerNormalization(
  [xOp, scaleOp, biasOp]: Operand[],
  { axis = -1, epsilon = 1e-5 }: { axis?: number; epsilon?: number },
): Operand[] {
  let x = operandToJax(xOp);
  const rank = x.ndim;
  if (axis < 0) axis += rank;

  // Normalize over axes [axis, ..., rank-1]
  const normAxes = Array.from({ length: rank - axis }, (_, i) => axis + i);

  const mean = x.ref.mean(normAxes, { keepdims: true });
  const diff = x.sub(mean);
  const variance = np.square(diff.ref).mean(normAxes, { keepdims: true });
  const invStd = np.reciprocal(np.sqrt(variance.add(epsilon)));
  x = diff.mul(invStd);

  if (scaleOp) {
    const scale = operandToJax(scaleOp);
    // scale shape is [features], need to broadcast to x shape
    const broadcastShape = [
      ...new Array(axis).fill(1),
      ...scale.shape,
    ];
    x = x.mul(scale.reshape(broadcastShape));
  }
  if (biasOp) {
    const bias = operandToJax(biasOp);
    const broadcastShape = [
      ...new Array(axis).fill(1),
      ...bias.shape,
    ];
    x = x.add(bias.reshape(broadcastShape));
  }

  return [x];
}

/**
 * InstanceNormalization (ONNX opset 6).
 *
 * Normalizes per (batch, channel) pair over spatial dimensions.
 *
 * Y[n,c,...] = (X[n,c,...] - mean) / sqrt(var + epsilon) * scale[c] + bias[c]
 */
export function InstanceNormalization(
  [xOp, scaleOp, biasOp]: Operand[],
  { epsilon = 1e-5 }: { epsilon?: number },
): Operand[] {
  const x = operandToJax(xOp);
  const [, C] = x.shape;
  const spatialAxes = Array.from({ length: x.ndim - 2 }, (_, i) => i + 2);

  const mean = x.ref.mean(spatialAxes, { keepdims: true });
  const diff = x.sub(mean);
  const variance = np.square(diff.ref).mean(spatialAxes, { keepdims: true });
  const invStd = np.reciprocal(np.sqrt(variance.add(epsilon)));
  let normed = diff.mul(invStd);

  if (scaleOp) {
    const scale = operandToJax(scaleOp);
    // scale: [C] → [1, C, 1, ...]
    const scaleShape = [1, C, ...spatialAxes.map(() => 1)];
    normed = normed.mul(scale.reshape(scaleShape));
  }
  if (biasOp) {
    const bias = operandToJax(biasOp);
    const biasShape = [1, C, ...spatialAxes.map(() => 1)];
    normed = normed.add(bias.reshape(biasShape));
  }

  return [normed];
}

export function ConstantOfShape(
  [input]: Operand[],
  { value }: { value?: TensorProto },
): Operand[] {
  const shape = operandToJs(input) as number[];
  if (value !== undefined) {
    const op = tensorToOperand(value);
    if (op instanceof StaticArray) {
      return [op.broadcastTo(shape)];
    } else {
      return [np.broadcastTo(op, shape)];
    }
  } else {
    return [np.zeros(shape)];
  }
}

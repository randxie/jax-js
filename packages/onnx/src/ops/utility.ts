// Utility operations, such as dtype conversion and data prep.
//
// TODO: OneHot (prompt_encoder_mask_decoder)
// TODO: ScatterND (prompt_encoder_mask_decoder)

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

export function Range([startOp, limitOp, deltaOp]: Operand[]): Operand[] {
  const start = Number(operandToJs(startOp));
  const limit = Number(operandToJs(limitOp));
  const delta = Number(operandToJs(deltaOp));
  const dtype = startOp.dtype;

  if (
    startOp instanceof StaticArray &&
    limitOp instanceof StaticArray &&
    deltaOp instanceof StaticArray &&
    dtype === np.int32
  ) {
    const values: number[] = [];
    if (delta > 0) {
      for (let x = start; x < limit; x += delta) values.push(x);
    } else if (delta < 0) {
      for (let x = start; x > limit; x += delta) values.push(x);
    } else {
      throw new Error("Range requires a non-zero delta");
    }
    return [new StaticArray(values, [values.length], dtype)];
  }

  return [np.arange(start, limit, delta, { dtype })];
}

export function LayerNormalization(
  [xOp, scaleOp, biasOp]: Operand[],
  { axis = -1, epsilon = 1e-5 }: { axis?: number; epsilon?: number },
): Operand[] {
  const x = operandToJax(xOp);
  const scale = operandToJax(scaleOp);
  const bias = biasOp ? operandToJax(biasOp) : null;

  const normalizedAxis = axis < 0 ? x.ndim + axis : axis;
  const reductionAxes = Array.from(
    { length: x.ndim - normalizedAxis },
    (_, i) => normalizedAxis + i,
  );
  const mean = x.ref.mean(reductionAxes, { keepdims: true });
  const centered = x.sub(mean);
  const variance = np.square(centered.ref).mean(reductionAxes, {
    keepdims: true,
  });
  let y = centered.div(np.sqrt(variance.add(epsilon)));
  y = y.mul(scale);
  return [bias ? y.add(bias) : y];
}

export function Pad(
  [xOp, padsOp, constantValueOp]: Operand[],
  { mode = "constant" }: { mode?: string },
): Operand[] {
  if (mode !== "constant") {
    throw new Error(`Pad mode is not supported: ${mode}`);
  }
  const x = operandToJax(xOp);
  const pads = operandToJs(padsOp) as number[];
  const rank = x.ndim;
  const width: Record<number, [number, number]> = {};
  for (let axis = 0; axis < rank; axis++) {
    width[axis] = [pads[axis], pads[axis + rank]];
  }

  const constantValue =
    constantValueOp === undefined ? 0 : Number(operandToJs(constantValueOp));
  if (constantValue !== 0) {
    throw new Error("Pad with non-zero constant value is not supported");
  }

  return [np.pad(x, width)];
}

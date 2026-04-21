import { expect, test } from "vitest";

import { fromNested, parse, toNested } from "./safetensors";

test("toNested() converts flat dictionary to nested object", () => {
  const b0 = 1;
  const w0 = 2;
  const b1 = 3;
  const w1 = 4;

  const flat = {
    "layers.0.bias": b0,
    "layers.0.weight": w0,
    "layers.1.bias": b1,
    "layers.1.weight": w1,
  };

  const nested = toNested(flat);

  expect(nested).toEqual({
    layers: [
      { bias: b0, weight: w0 },
      { bias: b1, weight: w1 },
    ],
  });
});

test("fromNested() converts nested object to flat dictionary", () => {
  const b0 = 100;
  const w0 = 200;
  const b1 = 312;
  const w1 = 434;

  const nested = {
    layers: [
      { bias: b0, weight: w0 },
      { bias: b1, weight: w1 },
    ],
  };

  const flat = fromNested(nested);

  expect(flat).toEqual({
    "layers.0.bias": b0,
    "layers.0.weight": w0,
    "layers.1.bias": b1,
    "layers.1.weight": w1,
  });
});

test("parse() loads BF16 tensors", () => {
  const header = JSON.stringify({
    tensor: {
      dtype: "BF16",
      shape: [2],
      data_offsets: [0, 4],
    },
  });
  const headerBytes = new TextEncoder().encode(header);
  const totalSize = 8 + headerBytes.length + 4;
  const bytes = new Uint8Array(totalSize);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  bytes.set(headerBytes, 8);
  const tensorBits = new Uint16Array(bytes.buffer, 8 + headerBytes.length, 2);
  tensorBits[0] = 0x3f80;
  tensorBits[1] = 0xc020;

  const file = parse(bytes);
  expect(file.tensors.tensor.dtype).toEqual("BF16");
  expect(Array.from(file.tensors.tensor.data as Uint16Array)).toEqual([0x3f80, 0xc020]);
});

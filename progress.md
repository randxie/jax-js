# FunASR Support Progress

Goal: support `https://github.com/FunAudioLLM/Fun-ASR` in `jax-js`, with the current checkpoint target being `FunAudioLLM/Fun-ASR-Nano-2512`.

## Current Status

- `frontend parity`: done
- `encoder + adaptor export to ONNX`: done
- `encoder + adaptor execution in jax-js ONNX runtime`: done
- `Qwen/FunASR prompt construction in JS`: done
- `Qwen tokenizer loading from local tokenizer.json`: done
- `Qwen/FunASR prompt tokenization parity`: done
- `FunASR source_ids / fbank_beg / fake_token_len preparation in JS`: done
- `Qwen weight export for JS runtime`: done
- `Qwen decoder implementation in JS`: in progress
- `browser-side WebGPU FunASR harness`: done
- `end-to-end transcript generation in jax-js`: in progress
- `VAD / long-audio segmentation`: not started
- `timestamps`: not started

## Verified Completed Work

- JS frontend parity exists in `scripts/funasr_nano_frontend.ts` and is checked by `scripts/verify_funasr_nano_frontend.ts`.
- Encoder + adaptor ONNX export exists in `scripts/export_funasr_nano_encoder.py`.
- Encoder + adaptor execution in `jax-js` exists in `scripts/run_funasr_nano_encoder.ts`.
- Python reference transcription exists in `scripts/verify_funasr_nano_transcription.py`.
- FunASR prompt construction exists in `scripts/funasr_nano_prompt.ts`.
- Hugging Face byte-level BPE loading exists in `packages/loaders/src/tokenizers.ts`.
- Qwen tokenizer parity verification exists in `scripts/verify_funasr_nano_tokenizer.ts`.
- FunASR prompt preparation exists in `scripts/funasr_nano_prepare.ts`.
- FunASR prompt preparation parity verification exists in `scripts/verify_funasr_nano_prepare.ts`.
- Qwen LLM export exists in `scripts/export_funasr_nano_llm.py`.
- Qwen runtime implementation exists in `scripts/funasr_nano_qwen.ts`.
- End-to-end runner exists in `scripts/verify_funasr_nano_e2e.ts`.
- Browser-side WebGPU route exists in `website/src/routes/funasr-nano/+page.svelte`.

## Latest Verification

Ran on the local assets already present in `.download/`:

- `npx pnpm exec tsx scripts/verify_funasr_nano_frontend.ts`
  - result: `shape=[1,94,560]`
  - result: `maxAbs=0.0002853870391845703`
  - result: `meanAbs=0.000005141351411634303`
- `npx pnpm exec tsx scripts/run_funasr_nano_encoder.ts`
  - result: `shape=[1,94,1024]`
  - result: `lens=[94]`
  - result: `maxAbs=0.001220703125`
  - result: `meanAbs=0.000057433418639467416`
- `.venv/bin/python scripts/verify_funasr_nano_transcription.py --expected-text 开饭时间早上九点至下午五点。`
  - result: transcript matches expected sample text
- `npx pnpm exec tsx scripts/verify_funasr_nano_tokenizer.ts`
  - result: local JS tokenizer matches the local `Qwen3-0.6B/tokenizer.json`
  - result: exact parity for the full FunASR chat prompt used for sample transcription
  - result: sample FunASR prompt length is `55` tokens
- `npx pnpm exec tsx scripts/verify_funasr_nano_prepare.ts`
  - result: local JS preparation matches Python reference `source_ids`
  - result: `fbank_beg=18`
  - result: `fake_token_len=12`
  - result: `source_ids_len=35`
- `.venv/bin/python scripts/export_funasr_nano_llm.py`
  - result: exported `310` tensors to `.artifacts/local/funasr_nano_llm_fp16.safetensors`
  - result: output size is `1192099840` bytes
- `npx pnpm --dir website build`
  - result: website build succeeds with the new `/funasr-nano` route
  - note: build emits pre-existing SvelteKit/Svelte warnings but exits successfully

## Main Findings

1. The ongoing work is currently only enough for:
   `waveform -> frontend features -> encoder/adaptor embeddings`

2. `FunASR-nano` transcription is not a small CTC-only decode path.
   The reference model builds:
   - audio encoder
   - audio adaptor
   - `Qwen3-0.6B` causal LM

3. The shipped local checkpoint does not contain CTC decoder weights.
   During the Python reference load, `ctc_decoder.*` and `ctc.*` keys are missing from `model.pt`.
   That means the working sample transcript comes from the LLM generation path, not from a lightweight CTC fallback.

4. Because of that, full `FunASR-nano` support in `jax-js` is still a substantial gap.
   The missing part is not frontend math anymore. The missing part is the decoder/runtime path around the Qwen model.

5. The tokenizer/prompt side is no longer speculative.
   `jax-js` can now:
   - build the FunASR chat prompt in JS
   - load the local `Qwen3-0.6B/tokenizer.json`
   - reproduce the exact token IDs for the sample FunASR prompt

6. The audio placeholder splice contract is now verified in JS.
   `jax-js` can now reproduce the exact Python-side:
   - `source_ids`
   - `fbank_beg`
   - `fake_token_len`

7. The repo now has a full JS-side decoder path on paper:
   - waveform -> JS frontend
   - ONNX encoder/adaptor in `jax-js`
   - prompt/source-id preparation in JS
   - audio embedding splice in JS
   - Qwen greedy decode implementation in JS

8. The remaining blocker is runtime feasibility, not missing wiring.
   Current attempts hit one or more of:
   - wasm `float16` op gaps
   - wasm allocator limits for larger `float32` loads
   - CPU-backend runtime that is too slow for practical end-to-end completion

9. Python greedy reference for the sample is now pinned:
   - generated token ids:
     `[29767, 99938, 20450, 105083, 99609, 27442, 56137, 102172, 75108, 27442, 1773, 151645]`
   - decoded text:
     `开饭时间早上九点至下午五点。`

10. WebGPU now has a browser-side entrypoint.
   The repo includes a `/funasr-nano` page that:
   - forces the `webgpu` backend
   - accepts local exported FunASR artifacts
   - runs frontend + encoder/adaptor + Qwen decode in-browser

11. The browser-side WebGPU route is staged but not transcript-verified in this shell environment.
   This environment does not expose `navigator.gpu`, so verification here is limited to a successful website build.

## Distance To FunASR-nano Support

Assessment: `not close yet` for full end-to-end support, but `good progress` on the audio side.

Roughly:

- audio preprocessing: mostly there
- acoustic encoder/adaptor path: mostly there
- text decoding path: missing
- production features like VAD and timestamps: missing

If the success bar is "match the sample transcript inside jax-js", the remaining work is dominated by the LLM side.

## Blocking Items

- [ ] Decide whether the target is:
  - full `FunASR-nano` support, or
  - a narrower milestone of `frontend + encoder/adaptor` support only
- [ ] Confirm whether we want to support the released `Fun-ASR-Nano-2512` checkpoint specifically
  - This matters because the released checkpoint does not provide a usable CTC path.
- [ ] Add a `jax-js` inference path for the Qwen decoder stack used by `FunASR-nano`
- [x] Load the tokenizer and special tokens used by the prompt/chat template
- [x] Recreate the prompt/chat template text used by FunASR inference
- [x] Recreate prompt-side fake audio token insertion metadata
- [x] Recreate `inputs_embeds` injection for audio tokens
- [x] Implement autoregressive generation and stopping rules
- [x] Add a browser-side `webgpu` execution path
- [ ] Verify that the browser-side `webgpu` path reproduces the sample transcript
- [ ] Add a verified end-to-end parity test against the sample audio transcript

## Shortest Practical Path

1. Freeze the current milestone as:
   `FunASR-nano audio frontend + encoder/adaptor parity`

2. Decide whether to continue with:
   - `FunASR-nano` as-is, which requires the Qwen generation path
   - another FunASR model that has a simpler decode path and is easier to bring up in `jax-js`

3. If staying with `FunASR-nano`, next implementation work should be:
   - export or load Qwen weights in a format usable from `jax-js`
   - export the Qwen token embedding table or otherwise load it in JS
   - use the new browser-side `webgpu` route to validate the decode path
   - then verify on the shipped sample audio

## Suggested Next Milestone

`Milestone: end-to-end FunASR-nano sample transcription in jax-js`

Definition of done:

- a single script takes the sample audio
- runs frontend in JS
- runs encoder/adaptor in `jax-js`
- runs the Qwen decode path in `jax-js`
- returns `开饭时间早上九点至下午五点。`

## Notes

- The local repo already contains the assets needed to keep validating the current audio-side work:
  - `.download/Fun-ASR/`
  - `.download/Fun-ASR-Nano-2512/`
- The broader `Fun-ASR` project includes VAD and timestamp-related features, but those are not the immediate blocker for `FunASR-nano` sample transcription.

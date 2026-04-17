import argparse
import json
import os
import sys

import torch
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the FunASR-nano encoder+adaptor path to ONNX.",
    )
    parser.add_argument(
        "--model-dir",
        default=".download/Fun-ASR-Nano-2512",
        help="Local FunASR-nano checkpoint directory.",
    )
    parser.add_argument(
        "--repo-dir",
        default=".download/Fun-ASR",
        help="Local Fun-ASR repository checkout containing model.py.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Audio file to featurize. Defaults to <model-dir>/example/zh.mp3.",
    )
    parser.add_argument(
        "--onnx-out",
        default=None,
        help="Output ONNX path. Defaults to <model-dir>/funasr_nano_encoder.onnx.",
    )
    parser.add_argument(
        "--sample-out",
        default=None,
        help="Output JSON path for the sample input/output pair.",
    )
    return parser.parse_args()


class EncoderExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, speech, speech_lengths):
        return self.model.forward_export(speech, speech_lengths)


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    repo_dir = os.path.abspath(args.repo_dir)
    remote_code = os.path.join(repo_dir, "model.py")
    audio_path = os.path.abspath(args.audio or os.path.join(model_dir, "example/zh.mp3"))
    onnx_out = os.path.abspath(
        args.onnx_out or os.path.join(model_dir, "funasr_nano_encoder.onnx")
    )
    sample_out = os.path.abspath(
        args.sample_out or os.path.join(model_dir, "funasr_nano_encoder_sample.json")
    )

    for path in (model_dir, repo_dir, remote_code, audio_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    sys.path.insert(0, repo_dir)
    from model import FunASRNano

    model, kwargs = FunASRNano.from_pretrained(
        model=model_dir,
        device="cpu",
        remote_code=remote_code,
    )
    model.eval()
    export_model = EncoderExport(model).eval()
    kwargs["frontend"].dither = 0.0

    audio = load_audio_text_image_video(audio_path, fs=16000, data_type="sound")
    speech, speech_lengths = extract_fbank(
        audio,
        data_type="sound",
        frontend=kwargs["frontend"],
    )
    speech_ref = speech.detach().clone()
    speech_lengths_ref = speech_lengths.detach().clone()
    audio_ref = audio.detach().clone()
    speech_export = speech_ref.clone()
    speech_lengths_export = speech_lengths_ref.clone()
    with torch.no_grad():
        encoder_out, encoder_out_lens = export_model(speech_export, speech_lengths_export)
    encoder_out_ref = encoder_out.detach().clone()
    encoder_out_lens_ref = encoder_out_lens.detach().clone()

    torch.onnx.export(
        export_model,
        (speech_ref.clone(), speech_lengths_ref.clone()),
        onnx_out,
        input_names=["speech", "speech_lengths"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "speech": {0: "batch", 1: "frames"},
            "speech_lengths": {0: "batch"},
            "encoder_out": {0: "batch", 1: "frames_out"},
            "encoder_out_lens": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    with open(sample_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "speech_shape": list(speech.shape),
                "speech_lengths": speech_lengths_ref.tolist(),
                "speech": speech_ref.flatten().tolist(),
                "waveform_length": int(audio_ref.shape[0]),
                "waveform": audio_ref.flatten().tolist(),
                "frontend": {
                    "fs": kwargs["frontend"].fs,
                    "window": kwargs["frontend"].window,
                    "n_mels": kwargs["frontend"].n_mels,
                    "frame_length": kwargs["frontend"].frame_length,
                    "frame_shift": kwargs["frontend"].frame_shift,
                    "lfr_m": kwargs["frontend"].lfr_m,
                    "lfr_n": kwargs["frontend"].lfr_n,
                    "dither": kwargs["frontend"].dither,
                    "snip_edges": kwargs["frontend"].snip_edges,
                    "upsacle_samples": kwargs["frontend"].upsacle_samples,
                },
                "encoder_out_shape": list(encoder_out_ref.shape),
                "encoder_out_lens": encoder_out_lens_ref.tolist(),
                "encoder_out": encoder_out_ref.flatten().tolist(),
            },
            f,
        )

    print(json.dumps({"onnx_out": onnx_out, "sample_out": sample_out}))


if __name__ == "__main__":
    main()

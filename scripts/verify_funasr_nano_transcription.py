import argparse
import json
import os
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with the local FunASR-nano checkpoint and optionally verify the transcript.",
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
        default=".download/Fun-ASR-Nano-2512/example/zh.mp3",
        help="Audio file to transcribe.",
    )
    parser.add_argument(
        "--expected-text",
        default=None,
        help="Optional expected transcript. Comparison uses a normalized form that ignores whitespace and punctuation.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u3000\u4e00-\u9fff]", "", text)


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    repo_dir = os.path.abspath(args.repo_dir)
    remote_code = os.path.join(repo_dir, "model.py")
    audio_path = os.path.abspath(args.audio)

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

    results, _ = model.inference(
        [audio_path],
        tokenizer=kwargs["tokenizer"],
        frontend=kwargs["frontend"],
        device="cpu",
    )
    transcript = results[0]["text"]

    output = {
        "audio": audio_path,
        "transcript": transcript,
        "normalized_transcript": normalize_text(transcript),
    }

    if args.expected_text is not None:
        expected_normalized = normalize_text(args.expected_text)
        output["expected_text"] = args.expected_text
        output["normalized_expected_text"] = expected_normalized
        output["matches"] = output["normalized_transcript"] == expected_normalized
        print(json.dumps(output, ensure_ascii=False))
        if not output["matches"]:
            raise SystemExit(1)
        return

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

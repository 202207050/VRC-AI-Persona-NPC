"""Compare greedy and beam 5 on CUDA. This script does not load the app's .env."""
import argparse
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persona_npc_app import Transcriber, now_ms, read_wav
from voice_options import hint_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="*", type=Path, help="16-bit PCM speech WAV, if available")
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL") or "turbo")
    parser.add_argument("--model-dir", type=Path, help="folder containing the existing Whisper checkpoint")
    args = parser.parse_args()

    import torch
    import whisper

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    root = args.model_dir or Path(os.environ.get("WHISPER_MODEL_DIR") or Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper")
    model_url = whisper._MODELS.get(args.model)
    if model_url and not (root / model_url.rsplit("/", 1)[-1]).is_file():
        raise SystemExit(f"Checkpoint is missing in {root}. Pass --model-dir to use the app's existing model.")
    os.environ["WHISPER_MODEL_DIR"] = str(root)
    os.environ["W04_WHISPER_BEAM_SIZE"] = "1"
    tr = Transcriber(args.model)
    if tr.device != "cuda":
        raise SystemExit("Whisper fell back to CPU.")
    print(f"{torch.cuda.get_device_name(0)} | Whisper {args.model}")

    prompt = hint_prompt(())
    paths = args.wav or sorted([*(ROOT / "tests").glob("*.wav"), *(ROOT / "clips").glob("*.wav")])
    if paths:
        for path in paths:
            audio = read_wav(path)

            def run(beam):
                tr.beam_size = beam
                tr.run(audio, initial_prompt=prompt)

            measure(run, str(path), torch)
    else:
        # Fixed decoder work on zero mel input, without synthesizing or recording speech.
        # EOT suppression makes each decoder execute exactly 16 or 32 steps.
        tokenizer = whisper.tokenizer.get_tokenizer(tr.model.is_multilingual, language="ko", task="transcribe")
        mel = torch.zeros((tr.model.dims.n_mels, whisper.audio.N_FRAMES), device="cuda")
        for steps in (16, 32):
            def run(beam):
                tr.model.decode(mel, whisper.DecodingOptions(
                    language="ko", task="transcribe", fp16=True, temperature=0.0, without_timestamps=True,
                    beam_size=beam, sample_len=steps, prompt=prompt, suppress_tokens=[-1, tokenizer.eot]))

            measure(run, f"encoder + {steps} decoder steps (proxy, no speech WAV)", torch)


def measure(run, label, torch):
    timings = {None: [], 5: []}
    for beam in timings:
        run(beam)
    for index in range(8):
        for beam in ((None, 5) if index % 2 == 0 else (5, None)):
            torch.cuda.synchronize()
            started = now_ms()
            run(beam)
            torch.cuda.synchronize()
            timings[beam].append(now_ms() - started)
    greedy, beam5 = (median(timings[beam]) for beam in timings)
    print(f"{label}: greedy {greedy:.1f} ms, beam 5 {beam5:.1f} ms, delta {beam5 - greedy:+.1f} ms (median, n=8)")


if __name__ == "__main__":
    main()

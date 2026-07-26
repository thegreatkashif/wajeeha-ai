"""Voice cloning for Wajeeha AI.

Runs entirely on the laptop's CPU (i7-11th gen / 16GB is plenty for
inference, though not real-time — expect a couple of seconds per sentence).

How it works:
  1. Drop 3-10 short WAV/MP3 clips of the target voice (clean, minimal
     background noise, ~10-30s each is enough — more helps) into
     `voice.reference_samples_dir` from config.yaml (default:
     ./voice_samples/).
  2. Flip `voice.enabled: true` in config.yaml.
  3. `VoiceCloner.synthesize(text)` clones the voice from those samples and
     returns a WAV file. Coqui's XTTS-v2 does this with no training step —
     the reference clips are used directly as a speaker conditioning
     signal at inference time, not baked into a fine-tuned model.

Model: Coqui XTTS-v2 (open weights, CPU-capable, multilingual). Pulled
automatically by the `TTS` package on first use.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path


class VoiceCloner:
    def __init__(
        self,
        reference_samples_dir: str,
        output_dir: str,
        language: str = "en",
    ) -> None:
        self._samples_dir = Path(reference_samples_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._language = language
        self._tts = None  # lazy-loaded — the model is ~1.8GB, don't pay
        # that cost (or the import cost) unless voice is actually used.

    def _reference_clips(self) -> list[str]:
        clips: list[str] = []
        for ext in ("*.wav", "*.mp3", "*.flac"):
            clips.extend(glob.glob(str(self._samples_dir / ext)))
        if not clips:
            raise FileNotFoundError(
                f"No reference audio found in {self._samples_dir}. Add a few "
                "clean WAV/MP3 clips of the target voice (10-30s each) and "
                "try again."
            )
        return clips

    def _load_model(self):
        if self._tts is None:
            from TTS.api import TTS  # heavy import — deferred on purpose

            # xtts_v2 is multilingual + zero-shot voice cloning from raw
            # reference audio, no fine-tuning/training required.
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._tts

    def synthesize(self, text: str, out_filename: str = "output.wav") -> str:
        """Clone the reference voice speaking `text`. Returns the output
        WAV path."""
        tts = self._load_model()
        speaker_wavs = self._reference_clips()
        out_path = self._output_dir / out_filename
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wavs,
            language=self._language,
            file_path=str(out_path),
        )
        return str(out_path)

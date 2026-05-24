"""
audio.py
--------
Streams microphone audio to Google Cloud Speech-to-Text and prints transcripts
in real time. Used as the audio-analysis component of the Attention Span
Detection system.

Prerequisites:
    pip install google-cloud-speech pyaudio six

Authentication:
    Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path
    of your Google Cloud service-account JSON key:

        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json

    See: https://cloud.google.com/docs/authentication/getting-started

Usage:
    python audio.py
"""

from __future__ import division

import os
import re
import sys
import time
from queue import Queue

import pyaudio
from google.cloud import speech

# ── Audio recording parameters ─────────────────────────────────
RATE = 16000
CHUNK = int(RATE / 10)   # 100 ms chunks
LANGUAGE_CODE = "en-US"
EXIT_PHRASES = re.compile(r"\b(exit|quit)\b", re.I)


class MicrophoneStream:
    """Opens a recording stream as a generator yielding raw audio chunks."""

    def __init__(self, rate: int = RATE, chunk: int = CHUNK) -> None:
        self._rate = rate
        self._chunk = chunk
        self._buff: Queue = Queue()
        self.closed = True

    def __enter__(self) -> "MicrophoneStream":
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Callback: push incoming audio data onto the queue."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Yield audio chunks from the queue until the stream closes."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except Exception:
                    break
            yield b"".join(data)


def listen_print_loop(responses) -> None:
    """
    Iterate over Speech-to-Text streaming responses and print transcripts.

    Exits when EXIT_PHRASES are detected or the stream ends.
    """
    num_chars_printed = 0

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            num_chars_printed = 0

            if EXIT_PHRASES.search(transcript):
                print("[INFO] Exit phrase detected — stopping.")
                break


def main() -> None:
    """Authenticate, open the mic stream, and transcribe until exit."""
    # Credentials are picked up automatically from the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS. Set it before running:
    #   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print(
            "[ERROR] GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "  Set it with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json",
            file=sys.stderr,
        )
        sys.exit(1)

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=LANGUAGE_CODE,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    timestamps = [time.time()]
    print(f"[INFO] Stream started at {timestamps[0]:.2f}")

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in audio_generator
            )
            responses = client.streaming_recognize(streaming_config, requests)
            listen_print_loop(responses)
    except Exception as exc:
        print(f"[WARN] Stream exceeded max duration or error occurred: {exc}")
        print("[INFO] Restarting stream...")
        main()


if __name__ == "__main__":
    main()

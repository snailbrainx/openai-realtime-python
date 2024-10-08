# openai_realtime/audio_utils.py

import sounddevice as sd
import numpy as np
import threading
import base64
import queue


def list_audio_devices():
    return sd.query_devices()


class AudioRecorder:
    """Class to handle audio recording from the microphone."""

    def __init__(self, input_device_name, samplerate=24000, block_duration=0.25):
        self.input_device_name = input_device_name
        self.samplerate = samplerate  # Fixed to 24kHz as required by OpenAI Realtime API
        self.blocksize = int(samplerate * block_duration)  # 0.25-second blocks
        self.channels = 1
        self.dtype = 'int16'
        self.audio_queue = queue.Queue()
        self.stop_flag = threading.Event()

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Recording Status: {status}")
        self.audio_queue.put(indata.copy())

    def record_audio(self):
        """Generator function to record audio chunks from the microphone."""
        with sd.InputStream(
            device=self.input_device_name,
            channels=self.channels,
            samplerate=self.samplerate,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._callback
        ):
            print(f"Recording started with samplerate: {self.samplerate} Hz")
            while not self.stop_flag.is_set():
                audio_chunk = self.audio_queue.get()
                yield audio_chunk.tobytes()

    def stop(self):
        self.stop_flag.set()


def encode_audio_chunk(audio_bytes):
    """Encode PCM audio bytes to base64 string."""
    return base64.b64encode(audio_bytes).decode('utf-8')


def decode_audio_chunk(encoded_audio):
    """Decode base64 string to PCM audio bytes."""
    return base64.b64decode(encoded_audio)


class AudioPlayer:
    """Class to handle audio playback to the speaker."""

    def __init__(self, output_device_name):
        self.output_device_name = output_device_name
        self.samplerate = 24000  # Fixed to 24kHz
        self.channels = 1
        self.buffer = np.array([], dtype='int16')
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
        self.total_samples_played = 0
        self.total_samples_played_lock = threading.Lock()
        self.playback_finished = threading.Event()
        self._create_stream()

    def is_playing(self):
        return not self.playback_finished.is_set()

    def _create_stream(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            blocksize=1024,
            device=self.output_device_name,
            channels=self.channels,
            dtype='int16',
            callback=self._callback,
            finished_callback=self._finished_callback
        )
        self.stream.start()

    def _callback(self, outdata, frames, time, status):
        if status:
            print(f"Playback Status: {status}")
        with self.buffer_lock:
            num_samples = frames * self.channels
            buffer_samples = len(self.buffer)
            samples_to_play = min(buffer_samples, num_samples)
            frames_to_play = samples_to_play // self.channels

            if samples_to_play > 0:
                data = self.buffer[:samples_to_play].reshape(-1, self.channels)
                outdata[:frames_to_play] = data
                self.buffer = self.buffer[samples_to_play:]
            else:
                frames_to_play = 0

            if frames_to_play < frames:
                outdata[frames_to_play:] = np.zeros((frames - frames_to_play, self.channels), dtype='int16')

            if len(self.buffer) == 0 and samples_to_play == 0:
                self.playback_finished.set()
            else:
                self.playback_finished.clear()

        samples_played = frames * self.channels
        with self.total_samples_played_lock:
            self.total_samples_played += samples_played

    def _finished_callback(self):
        pass

    def stop(self):
        if self.stream.active:
            self.stream.stop()
            self.stream.close()
        with self.buffer_lock:
            self.buffer = np.array([], dtype='int16')
        self.playback_finished.set()

    def reset(self):
        if self.stream.active:
            self.stream.stop()
            self.stream.close()
        self._create_stream()
        with self.buffer_lock:
            self.buffer = np.array([], dtype='int16')
        with self.total_samples_played_lock:
            self.total_samples_played = 0
        self.playback_finished.clear()

    def write(self, audio_bytes):
        with self.buffer_lock:
            audio_array = np.frombuffer(audio_bytes, dtype='<i2')  # Little-endian int16
            self.buffer = np.concatenate((self.buffer, audio_array))
        self.playback_finished.clear()

    def get_total_played_samples(self):
        with self.total_samples_played_lock:
            print(f"Total samples played retrieved: {self.total_samples_played}")
            return self.total_samples_played

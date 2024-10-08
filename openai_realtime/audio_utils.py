# openai_realtime/audio_utils.py

import sounddevice as sd
import numpy as np
import threading
import base64, queue

def list_audio_devices():
    devices = sd.query_devices()
    return devices

def record_audio(input_device_name):
    """Generator function to record audio chunks from the microphone."""
    def callback(indata, frames, time, status):
        if status:
            print(f"Recording Status: {status}")
        audio_queue.put(indata.copy())
    
    audio_queue = queue.Queue()
    samplerate = 24000  # Fixed to 24kHz as required by OpenAI Realtime API

    with sd.InputStream(
        device=input_device_name,
        channels=1,
        samplerate=samplerate,
        dtype='int16',
        blocksize=int(samplerate * 0.25),  # 0.25-second blocks
        callback=callback
    ):
        print(f"Recording started with samplerate: {samplerate} Hz")
        while True:
            audio_chunk = audio_queue.get()
            yield audio_chunk.tobytes()

def encode_audio_chunk(audio_bytes):
    """Encode PCM audio bytes to base64 string."""
    return base64.b64encode(audio_bytes).decode('utf-8')

def decode_audio_chunk(encoded_audio):
    """Decode base64 string to PCM audio bytes."""
    return base64.b64decode(encoded_audio)

class AudioPlayer:
    def __init__(self, output_device_name):
        self.output_device_name = output_device_name
        self.samplerate = 24000  # Fixed to 24kHz
        self.channels = 1
        self.buffer = np.array([], dtype='int16')
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()  # Lock for thread safety
        self.total_samples_played = 0
        self.total_samples_played_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.playback_finished = threading.Event()  # Add this line
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
            callback=self.callback,
            finished_callback=self.finished_callback
        )
        self.stream.start()

    def callback(self, outdata, frames, time, status):
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
                # Buffer is empty and we have no samples to play
                self.playback_finished.set()  # Set the event when playback is done
            else:
                self.playback_finished.clear()  # Clear the event when there is data to play

        samples_played = frames * self.channels 

        with self.total_samples_played_lock:
            self.total_samples_played += samples_played

    def finished_callback(self):
        pass

    def stop(self):
        with self.stream_lock:
            if self.stream.active:
                self.stream.stop()
                self.stream.close()
        with self.buffer_lock:
            self.buffer = np.array([], dtype='int16')
        self.playback_finished.set()  # Indicate that playback has stopped

    def reset(self):
        with self.stream_lock:
            if self.stream.active:
                self.stream.stop()
                self.stream.close()
            self._create_stream()
        with self.buffer_lock:
            self.buffer = np.array([], dtype='int16')
        with self.total_samples_played_lock:
            self.total_samples_played = 0  # Ensure this is reset
            self.stop_flag.clear()
        self.playback_finished.clear()  # Ensure this is reset

    def write(self, audio_bytes):
        with self.buffer_lock:
            audio_array = np.frombuffer(audio_bytes, dtype='<i2')  # Little-endian int16
            self.buffer = np.concatenate((self.buffer, audio_array))
        self.playback_finished.clear()  # Clear the event when new data is written

    def get_total_played_samples(self):
        with self.total_samples_played_lock:
            print(f"Total samples played retrieved: {self.total_samples_played}")
            return self.total_samples_played

# main.py

import os
import json
from openai_realtime.client import RealtimeClient

SETTINGS_FILE = 'settings.json'


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)


def select_audio_devices():
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    output_devices = [d for d in devices if d['max_output_channels'] > 0]

    print("\nAvailable Input Devices:")
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']}")

    selected_input_device = input_devices[_select_device(len(input_devices), "Input")]['name']

    print("\nAvailable Output Devices:")
    for idx, dev in enumerate(output_devices):
        print(f"{idx}: {dev['name']}")

    selected_output_device = output_devices[_select_device(len(output_devices), "Output")]['name']

    # Voice selection
    voices = ['alloy', 'echo', 'shimmer']  # Update with actual voice options
    print("\nAvailable Voices:")
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice}")

    selected_voice = voices[_select_device(len(voices), "Voice")]

    settings = {
        'input_device': selected_input_device,
        'output_device': selected_output_device,
        'voice': selected_voice
    }
    save_settings(settings)
    print("Audio settings saved.")
    return settings


def _select_device(max_index, device_type):
    while True:
        try:
            idx = int(input(f"Select {device_type} Device by index: "))
            if 0 <= idx < max_index:
                return idx
            else:
                print("Invalid index. Please enter a valid index.")
        except ValueError:
            print("Invalid input. Please enter a valid index.")


def main():
    settings = load_settings()
    if settings:
        change = input("Do you want to change your audio settings? Y/N: ").strip().lower()
        if change == 'y':
            settings = select_audio_devices()
        else:
            print("Using saved audio settings.")
    else:
        settings = select_audio_devices()

    client = RealtimeClient(settings)
    client.connect()

    print("\nYou can start speaking to the assistant. Say 'exit' or press Ctrl+C to quit.")

    client.start_audio_stream()


if __name__ == "__main__":
    main()

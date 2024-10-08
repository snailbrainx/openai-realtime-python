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

    while True:
        try:
            input_idx = int(input("Select Input Device by index: "))
            selected_input_device = input_devices[input_idx]['name']
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid index.")

    print("\nAvailable Output Devices:")
    for idx, dev in enumerate(output_devices):
        print(f"{idx}: {dev['name']}")

    while True:
        try:
            output_idx = int(input("Select Output Device by index: "))
            selected_output_device = output_devices[output_idx]['name']
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid index.")

    # Voice selection
    voices = ['alloy', 'echo', 'shimmer']  # Update with actual voice options
    print("\nAvailable Voices:")
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice}")

    while True:
        try:
            voice_idx = int(input("Select Voice by index: "))
            selected_voice = voices[voice_idx]
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid index.")

    settings = {
        'input_device': selected_input_device,
        'output_device': selected_output_device,
        'voice': selected_voice
    }
    save_settings(settings)
    print("Audio settings saved.")
    return settings

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

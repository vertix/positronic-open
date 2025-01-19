import os

import ironic as ir

def add_notification(port: ir.OutputPort, sound_name: str, sound_dir: str = 'assets/sounds'):
    import playsound

    sound_path = os.path.join(sound_dir, f"{sound_name}.wav")

    async def play_sound(message: ir.Message):
        playsound.playsound(sound_path, block=False)

    port.subscribe(play_sound)

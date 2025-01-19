import ironic as ir

def add_notification(port: ir.OutputPort, sound_path: str):
    import playsound

    async def play_sound(message: ir.Message):
        playsound.playsound(sound_path, block=False)

    port.subscribe(play_sound)

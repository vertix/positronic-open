from webxr import WebXR
from control import utils


def main():
    webxr = WebXR(port=5005)
    # logger = utils.Logger(inputs=["transform", "buttons"])
    logger = utils.Map(inputs=["transform", "buttons"], default=lambda n, v : print(f'{n}: {v}'))
    logger.ins.transform = webxr.outs.transform
    logger.ins.buttons = webxr.outs.buttons
    webxr.start()  # WebXR must start last, as it is a blocking call
    logger.start()

    next(webxr._control_loop)


if __name__ == "__main__":
    main()

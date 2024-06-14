from flask import Flask, request, jsonify

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/track', methods=['POST'])
def track():
    data = request.json
    position = {k: f'{" " if v >= 0 else ""}{v:.3f}' for k, v in data['position'].items()}
    orientation = {k: f'{" " if v >= 0 else ""}{v:.3f}' for k, v in data['orientation'].items()}
    buttons = ''.join([' X'[b] for b in data['buttons']])
    timestamp = data['timestamp']

    print(f"Timestamp: {timestamp}, Position: {position}, Orientation: {orientation}, Buttons: {buttons}")

    # # Example of moving the robot based on the received position
    # # This needs to be adapted to your specific control logic
    # arm.move_to_position([position['x'], position['y'], position['z']])

    # # Example of handling button presses to control the gripper
    # if buttons[0]:  # Assuming the first button controls the gripper
    #     arm.gripper.close()
    # else:
    #     arm.gripper.open()

    return jsonify(success=True)

if __name__ == '__main__':
    # Please create your own cert.pem and key.pem files in the same directory as this script
    # You can generate them using openssl:
    # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
    app.run(host='0.0.0.0', port=5005, ssl_context=('cert.pem', 'key.pem'))

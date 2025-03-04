# ar_modules/socket_handlers.py
from ar_modules.image_processing import decode_image, encode_image, process_frame
import cv2

# Shared state variables
prev_gray = None
prev_points = None

def configure_socket_handlers(socketio):
    @socketio.on('input_frame')
    def handle_input_frame(data_url):
        global prev_gray, prev_points
        
        # Decode the incoming image
        frame = decode_image(data_url)
        if frame is None:
            return
            
        # Process the frame and get the result
        final_img, prev_gray, prev_points = process_frame(
            frame, prev_gray, prev_points)
            
        # Encode and emit the processed image
        encoded_final = encode_image(final_img)
        if encoded_final:
            socketio.emit('output_final', encoded_final)
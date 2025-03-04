# main.py
from flask import Flask
from flask_socketio import SocketIO
from ar_modules.routes import configure_routes
from ar_modules.socket_handlers import configure_socket_handlers

def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app)
    
    # Configure routes and socket handlers
    configure_routes(app)
    configure_socket_handlers(socketio)
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
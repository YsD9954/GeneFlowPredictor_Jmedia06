from flask import Flask
from flask_cors import CORS
import api  # Import the API blueprint

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
app.register_blueprint(api.bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

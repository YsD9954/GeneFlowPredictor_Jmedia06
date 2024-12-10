from flask import Blueprint, jsonify, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from Bio import SeqIO
from scipy.stats import norm
import torch
import torch.nn as nn

# Blueprint for API routes
bp = Blueprint('api', __name__)

# Neural network for trait propagation
class GeneFlowNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneFlowNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Load pre-trained model or initialize
gene_flow_model = GeneFlowNN(input_dim=3, output_dim=1)
gene_flow_model.eval()

@bp.route('/')
def home():
    return jsonify({
        "message": "Welcome to GeneFlow Predictor™ API",
        "endpoints": {
            "/api/uploadGenomicData": "POST - Upload and preprocess genetic data",
            "/api/uploadEnvData": "POST - Upload and preprocess environmental data",
            "/api/runSimulation": "POST - Simulate gene flow and predict outcomes",
            "/api/generateHeatmap": "POST - Generate a trait distribution heatmap"
        }
    })

# Upload genomic data
@bp.route('/api/uploadGenomicData', methods=['POST'])
def upload_genomic_data():
    file = request.files.get('genetic_data.fasta')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        sequences = list(SeqIO.parse(file, "fasta"))
        sequence_lengths = [len(seq.seq) for seq in sequences]
        avg_length = np.mean(sequence_lengths)
        return jsonify({
            "message": "Genomic data processed successfully",
            "num_sequences": len(sequences),
            "avg_sequence_length": avg_length
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Upload environmental data
@bp.route('/api/uploadEnvData', methods=['POST'])
def upload_env_data():
    file = request.files.get('environmental_data.csv')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        data = pd.read_csv(file)
        summary = data.describe().to_dict()
        return jsonify({"message": "Environmental data processed successfully", "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simulate gene flow
@bp.route('/api/runSimulation', methods=['POST'])
def run_simulation():
    payload = request.json
    mutation_rate = payload.get('mutation_rate', 0.01)
    migration_rate = payload.get('migration_rate', 0.05)
    selection_pressure = payload.get('selection_pressure', 1.0)

    try:
        # Simulated population genetics model
        generations = 100
        gene_frequencies = np.zeros(generations)
        gene_frequencies[0] = 0.5  # Initial frequency
        
        for t in range(1, generations):
            gene_frequencies[t] = (
                gene_frequencies[t-1] * (1 - mutation_rate) +
                migration_rate * (1 - gene_frequencies[t-1]) +
                selection_pressure * gene_frequencies[t-1] * (1 - gene_frequencies[t-1])
            )
        
        return jsonify({
            "message": "Simulation completed successfully",
            "gene_frequencies": gene_frequencies.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Generate heatmap
@bp.route('/api/generateHeatmap', methods=['POST'])
def generate_heatmap():
    try:
        # Generate random heatmap data
        data = np.random.rand(10, 10)
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Trait Distribution Heatmap")

        # Save heatmap to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_buffer.seek(0)

        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

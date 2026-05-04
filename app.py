"""
Nicasia CyberIntel — AI Prediction Service
PyTorch-based risk classifier for threats and incidents.
Model is persisted to disk so it is NOT retrained on every restart.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import mysql.connector
import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai-service.log")
    ]
)
logger = logging.getLogger("cyber-intel-ai")

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pt"

# ── Neural Network ────────────────────────────────────────────────────────────

class RiskPredictor(nn.Module):
    """
    Feed-forward network for cybersecurity risk classification.
    Architecture: 4 inputs → 32 hidden → 16 hidden → 4 outputs (LOW/MEDIUM/HIGH/CRITICAL)
    Upgraded from original: wider hidden layers + BatchNorm for stability.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.net(x)


# ── Encoders ─────────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    'Ransomware': 1.0, 'Data Breach': 0.95, 'DDoS': 0.75,
    'Trojan': 0.7, 'Worm': 0.6, 'Spyware': 0.5,
    'Phishing': 0.4, 'Virus': 0.35, 'Adware': 0.2
}

SOURCE_MAP = {
    'External': 1.0, 'Unknown': 0.7, 'Internal': 0.4, 'Vendor': 0.3
}

SEVERITY_MAP = {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.75, 'CRITICAL': 1.0}
STATUS_MAP   = {'OPEN': 1.0, 'IN_PROGRESS': 0.5, 'RESOLVED': 0.0}
LABELS       = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


def encode_threat(category: str, source: str, frequency: int) -> list:
    return [
        CATEGORY_MAP.get(category, 0.5),
        SOURCE_MAP.get(source, 0.5),
        min(frequency / 10.0, 1.0),
        1.0   # bias
    ]


def encode_incident(incident: dict) -> list:
    return [
        SEVERITY_MAP.get(incident.get('severity', 'LOW'), 0.5),
        CATEGORY_MAP.get(incident.get('type', ''), 0.5),
        STATUS_MAP.get(incident.get('status', 'OPEN'), 0.5),
        1.0
    ]


# ── Training ──────────────────────────────────────────────────────────────────

TRAINING_DATA = [
    # [category_enc, source_enc, freq_norm, bias] → label_idx
    ([1.0,  1.0, 0.8, 1.0], 3),   # Ransomware + External + high freq → CRITICAL
    ([0.95, 1.0, 0.5, 1.0], 3),   # Data Breach + External → CRITICAL
    ([0.75, 1.0, 0.6, 1.0], 2),   # DDoS + External → HIGH
    ([0.7,  0.7, 0.4, 1.0], 2),   # Trojan + Unknown → HIGH
    ([0.6,  0.4, 0.3, 1.0], 1),   # Worm + Internal → MEDIUM
    ([0.5,  0.4, 0.3, 1.0], 1),   # Spyware + Internal → MEDIUM
    ([0.4,  0.4, 0.2, 1.0], 1),   # Phishing + Internal → MEDIUM
    ([0.35, 0.3, 0.1, 1.0], 0),   # Virus + Vendor → LOW
    ([0.2,  0.3, 0.0, 1.0], 0),   # Adware + Vendor → LOW
    ([1.0,  0.7, 0.9, 1.0], 3),   # Ransomware + Unknown + very high → CRITICAL
    ([0.95, 0.7, 0.7, 1.0], 3),   # Data Breach + Unknown → CRITICAL
    ([0.75, 0.4, 0.2, 1.0], 1),   # DDoS + Internal → MEDIUM
    ([0.7,  1.0, 0.8, 1.0], 2),   # Trojan + External → HIGH
    ([0.35, 0.3, 0.5, 1.0], 0),   # Virus + Vendor + many → LOW
    ([0.4,  1.0, 0.6, 1.0], 1),   # Phishing + External + many → MEDIUM
]


def train_model(model: RiskPredictor):
    """Train model on synthetic representative data and return final loss."""
    X = torch.tensor([d[0] for d in TRAINING_DATA], dtype=torch.float32)
    y = torch.tensor([d[1] for d in TRAINING_DATA], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    final_loss = loss.item()

    # Compute training accuracy
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        accuracy = (preds == y).float().mean().item() * 100

    logger.info(f"Training complete. Loss: {final_loss:.4f} | Accuracy: {accuracy:.1f}%")
    return final_loss


def load_or_train_model() -> RiskPredictor:
    """Load persisted model from disk; train a fresh one if not found."""
    model = RiskPredictor()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            model.eval()
            logger.info(f"Model loaded from {MODEL_PATH}")
            return model
        except Exception as e:
            logger.warning(f"Could not load model ({e}), retraining...")

    logger.info("Training new model...")
    train_model(model)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    return model


model = load_or_train_model()


# ── Database ─────────────────────────────────────────────────────────────────

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "railway")
        ssl_disabled=False,
        connection_timeout=10
    )


def get_incidents_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, severity, type, status FROM incidents")
    incidents = cursor.fetchall()
    cursor.close()
    conn.close()
    return incidents


def get_threats_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, category, source, risk_level FROM threats")
    threats = cursor.fetchall()
    cursor.close()
    conn.close()
    return threats


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model': 'RiskPredictor v3',
        'model_persisted': os.path.exists(MODEL_PATH),
        'training_samples': len(TRAINING_DATA)
    })


@app.route('/predict-threat', methods=['POST'])
def predict_threat():
    """
    Called by Spring Boot AiService on every Threat create/update.
    Body: { "category": "Ransomware", "source": "External", "frequency": 3 }
    Returns: { "riskLevel": "HIGH", "confidence": 0.87 }
    """
    try:
        data = request.get_json(force=True)
        category  = data.get('category', '')
        source    = data.get('source', 'Unknown')
        frequency = int(data.get('frequency', 0))

        features = torch.tensor([encode_threat(category, source, frequency)], dtype=torch.float32)

        with torch.no_grad():
            logits = model(features)
            probs  = torch.softmax(logits, dim=1)
            idx    = torch.argmax(probs, dim=1).item()
            conf   = probs[0][idx].item()

        result = {
            'riskLevel':  LABELS[idx],
            'confidence': round(conf, 3)
        }
        logger.info(f"Threat prediction: {category}/{source} → {result['riskLevel']} ({conf:.2%})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Threat prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['GET'])
def predict_incidents():
    """
    Called by Angular admin panel to get AI predictions for all incidents.
    Returns one prediction per incident from the database.
    """
    try:
        incidents = get_incidents_from_db()
        results = []

        recommendations = {
            'CRITICAL': 'Immediate isolation and incident response team activation required',
            'HIGH':     'Escalate to security team — action required within 1 hour',
            'MEDIUM':   'Monitor closely and schedule patch/remediation within 24 hours',
            'LOW':      'Log and schedule routine security review'
        }

        for inc in incidents:
            features = torch.tensor([encode_incident(inc)], dtype=torch.float32)

            with torch.no_grad():
                logits = model(features)
                probs  = torch.softmax(logits, dim=1)
                idx    = torch.argmax(probs, dim=1).item()
                conf   = probs[0][idx].item()
                risk_score = round(conf * 100)

            predicted = LABELS[idx]
            results.append({
                'incidentId':        inc['id'],
                'title':             inc['title'],
                'currentSeverity':   inc['severity'],
                'predictedSeverity': predicted,
                'riskScore':         risk_score,
                'confidence':        round(conf, 3),
                'recommendation':    recommendations[predicted],
                'severityMatch':     inc['severity'] == predicted
            })

        results.sort(key=lambda r: r['riskScore'], reverse=True)
        logger.info(f"Batch prediction complete: {len(results)} incidents processed")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-incident', methods=['POST'])
def predict_single_incident():
    """
    Predict risk for a single incident payload (used for real-time form validation).
    Body: { "severity": "HIGH", "type": "Ransomware", "status": "OPEN" }
    """
    try:
        data = request.get_json(force=True)
        features = torch.tensor([encode_incident(data)], dtype=torch.float32)

        with torch.no_grad():
            logits = model(features)
            probs  = torch.softmax(logits, dim=1)
            idx    = torch.argmax(probs, dim=1).item()
            conf   = probs[0][idx].item()

        all_probs = {LABELS[i]: round(probs[0][i].item(), 3) for i in range(4)}

        return jsonify({
            'predictedSeverity': LABELS[idx],
            'confidence':        round(conf, 3),
            'riskScore':         round(conf * 100),
            'probabilities':     all_probs
        })

    except Exception as e:
        logger.error(f"Single incident prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Force retrain and persist the model.
    Protected by a simple API key for safety.
    """
    api_key = request.headers.get('X-API-Key', '')
    if api_key != os.getenv('AI_ADMIN_KEY', 'nicasia-ai-admin-2024'):
        return jsonify({'error': 'Unauthorised'}), 401

    global model
    model = RiskPredictor()
    loss = train_model(model)
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model retrained and saved via /retrain endpoint")
    return jsonify({'message': 'Model retrained successfully', 'loss': round(loss, 4)})


@app.route('/model-info')
def model_info():
    """Return model metadata for the admin dashboard."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return jsonify({
        'architecture':     'RiskPredictor v3 (4→32→16→4)',
        'totalParameters':  total_params,
        'trainableParams':  trainable,
        'trainingsamples':  len(TRAINING_DATA),
        'labels':           LABELS,
        'modelPersisted':   os.path.exists(MODEL_PATH)
    })


if __name__ == '__main__':
    port = int(os.getenv('AI_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f"Starting AI service on port {port}")
    app.run(port=port, debug=debug)

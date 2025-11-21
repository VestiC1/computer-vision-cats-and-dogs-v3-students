"""
═══════════════════════════════════════════════════════════════════════════════
🎯 PROMETHEUS METRICS - Export de métriques MLOps
═══════════════════════════════════════════════════════════════════════════════
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import os


def setup_prometheus(app):
    """
    Configure Prometheus pour FastAPI
    Compatible avec l'API existante V2

    Args:
        app: Instance FastAPI
    """
    if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        print("✅ Prometheus metrics enabled at /metrics")
        
    else:
        print("ℹ️  Prometheus metrics disabled")


# Créer métrique histogram pour latence
inference_time_histogram = Histogram(
    'cv_inference_time_seconds',
    'Temps d\'inférence en secondes'
)

# Feedback utilisateur
feedback_counter = Counter(
    'cv_feedback_total',
    'Nombre total de feedbacks utilisateurs',
    ['type']  # "positive" ou "negative"
)

# Répartition des prédictions
prediction_counter = Counter(
    'cv_prediction_total',
    'Nombre total de prédictions par label',
    ['label']  # "cat" ou "dog"
)

# Nombre total d'utilisations
usage_counter = Counter(
    'cv_usage_total',
    'Nombre total d\'utilisations de l\'API'
)

# Gauge pour le statut de la base de données
database_status_gauge = Gauge(
    'cv_database_connected',
    'Statut de la connexion à la base de données (1=connecté, 0=déconnecté)'
)

# Fonction : Track inference time
def track_inference_time(inference_time_ms: float):
    """Enregistre le temps d'inférence"""
    inference_time_histogram.observe(inference_time_ms / 1000)

# Fonction : Track feedback
def track_feedback(feedback_type: str):
    """Enregistre un feedback utilisateur"""
    feedback_counter.labels(type=feedback_type).inc()

# Fonction : Track prédiction
def track_prediction(label: str):
    """Enregistre une prédiction avec son label"""
    prediction_counter.labels(label=label).inc()
    usage_counter.inc()  # Incrémente le compteur global

# Fonction update db status
def update_db_status(connected: bool):
    """Met à jour le statut de la base de données"""
    database_status_gauge.set(1 if connected else 0)

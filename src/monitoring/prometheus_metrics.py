"""
═══════════════════════════════════════════════════════════════════════════════
🎯 PROMETHEUS METRICS - Export de métriques MLOps
═══════════════════════════════════════════════════════════════════════════════
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import os


database_status = Gauge(
    'cv_database_connected',
    'Database connection status (1=connected, 0=disconnected)'
)

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

def update_db_status(is_connected: bool):
    """
    Met à jour le statut de la base de données
    """
    database_status.set(1 if is_connected else 0)

# Créer métrique histogram pour latence
inference_time_histogram = Histogram(
    'cv_inference_time_seconds',
    'Temps d\'inférence en secondes'
)

def track_inference_time(inference_time_ms: float):
    """Enregistre le temps d'inférence"""
    inference_time_histogram.observe(inference_time_ms / 1000)

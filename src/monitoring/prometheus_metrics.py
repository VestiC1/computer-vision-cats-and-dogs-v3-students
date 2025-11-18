"""
═══════════════════════════════════════════════════════════════════════════════
🎯 PROMETHEUS METRICS - Export de métriques MLOps
═══════════════════════════════════════════════════════════════════════════════

📚 OBJECTIF PÉDAGOGIQUE
Ce module expose les métriques métier de l'application au format Prometheus.
Il illustre comment instrumenter une application ML pour le monitoring production.

🔑 CONCEPTS CLÉS
- Types de métriques Prometheus : Counter, Gauge, Histogram
- Instrumentation automatique vs manuelle (FastAPI)
- Labels pour dimensions multiples (segmentation des données)
- Buckets pour histogrammes (distribution des valeurs)

🔗 INTÉGRATION
- Appelé par : src/api/main.py (setup au démarrage)
- Consommé par : Prometheus (scrape /metrics toutes les 15s)
- Compatible V2 : s'ajoute au monitoring Plotly existant (complémentaire)

═══════════════════════════════════════════════════════════════════════════════
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import os

# ═══════════════════════════════════════════════════════════════════════════
# 📊 MÉTRIQUES CUSTOM - Spécifiques au modèle CV cats/dogs
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 📏 GAUGE : Valeur pouvant monter ET descendre (snapshot de l'état actuel)
# ─────────────────────────────────────────────────────────────────────────────
database_status = Gauge(
    'cv_database_connected',
    'Database connection status (1=connected, 0=disconnected)'
)
# 💡 USAGE
# - .set(1) : marque comme connecté
# - .set(0) : marque comme déconnecté
#
# 🎯 CAS D'USAGE
# - Monitoring santé infrastructure (alerte si = 0)
# - Corrélation : échecs prédictions ↔ base déconnectée ?
#
# 📈 QUERY PROMQL POUR ALERTE
# - cv_database_connected == 0 : déclenche alerte Discord

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 SETUP - Configuration de l'instrumentation Prometheus
# ═══════════════════════════════════════════════════════════════════════════
def setup_prometheus(app):
    """
    Configure Prometheus pour FastAPI
    Compatible avec l'API existante V2
    
    🎯 INSTRUMENTATION AUTOMATIQUE
    Le Instrumentator ajoute automatiquement :
    - http_request_duration_seconds : latence par endpoint
    - http_requests_total : nombre de requêtes par status code
    - http_requests_in_progress : requêtes concurrentes
    
    💡 ENDPOINT /metrics
    Exposé automatiquement au format Prometheus :
    # HELP cv_predictions_total Total number of predictions
    # TYPE cv_predictions_total counter
    cv_predictions_total{result="cat"} 42.0
    cv_predictions_total{result="dog"} 38.0
    
    Args:
        app: Instance FastAPI
    """
    if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
        # 📊 INSTRUMENTATION EN 2 ÉTAPES
        # 1. instrument(app) : ajoute middleware pour métriques auto
        # 2. expose(app, endpoint="/metrics") : crée route GET /metrics
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        print("✅ Prometheus metrics enabled at /metrics")
        
        # 💡 FORMAT DE SORTIE /metrics
        # Texte brut (Content-Type: text/plain)
        # Scrapable par Prometheus toutes les 15s (cf. prometheus.yml)
    else:
        print("ℹ️  Prometheus metrics disabled")
        # Utile en dev si on veut alléger le monitoring

# ═══════════════════════════════════════════════════════════════════════════
# 📝 HELPERS - Fonctions de tracking appelées par l'API
# ═══════════════════════════════════════════════════════════════════════════

def update_db_status(is_connected: bool):
    """
    Met à jour le statut de la base de données
    
    🔗 APPELÉ PAR : healthcheck ou retry logic de connexion DB
    
    Args:
        is_connected: True si connexion PostgreSQL active
    
    💡 EXEMPLE D'INTÉGRATION
    try:
        db.execute("SELECT 1")
        update_db_status(True)
    except Exception:
        update_db_status(False)
        # Alerte Grafana se déclenche automatiquement
    """
    database_status.set(1 if is_connected else 0)

# ═══════════════════════════════════════════════════════════════════════════
# 🎓 CONCEPTS AVANCÉS (pour aller plus loin)
# ═══════════════════════════════════════════════════════════════════════════
#
# 1. MÉTRIQUES SUPPLÉMENTAIRES UTILES
#    - model_version (Gauge avec label 'version') : tracking déploiements
#    - input_image_size (Histogram) : détection images hors distribution
#    - gpu_memory_usage (Gauge) : monitoring ressources (si GPU disponible)
#
# 2. CARDINALITY (nombre de combinaisons de labels)
#    ⚠️ Attention : trop de labels = explosion mémoire Prometheus
#    Exemple à ÉVITER : .labels(user_id=...) avec 1M users
#    Limite raisonnable : <10 valeurs par label
#
# 3. MÉTRIQUES VS LOGS
#    - Métriques : agrégées, numériques, queryable (dashboards, alertes)
#    - Logs : détaillés, textuels, debugging (ex: traceback erreurs)
#    Les deux sont complémentaires (pas l'un OU l'autre)
#
# 4. TESTS DES MÉTRIQUES
#    import pytest
#    def test_track_prediction():
#        before = predictions_total._value.get()
#        track_prediction('cat', 100, 0.95)
#        assert predictions_total._value.get() == before + 1
#
# ═══════════════════════════════════════════════════════════════════════════
# 📚 RESSOURCES PÉDAGOGIQUES
# ═══════════════════════════════════════════════════════════════════════════
#
# - Prometheus best practices: https://prometheus.io/docs/practices/naming/
# - Types de métriques expliqués: https://prometheus.io/docs/concepts/metric_types/
# - PromQL tutorial: https://prometheus.io/docs/prometheus/latest/querying/basics/
# - FastAPI Instrumentator: https://github.com/trallnag/prometheus-fastapi-instrumentator
#
# ═══════════════════════════════════════════════════════════════════════════

# Créer métrique histogram pour latence
inference_time_histogram = Histogram(
    'cv_inference_time_seconds',
    'Temps d\'inférence en secondes'
)

def track_inference_time(inference_time_ms: float):
    """Enregistre le temps d'inférence"""
    inference_time_histogram.observe(inference_time_ms / 1000)
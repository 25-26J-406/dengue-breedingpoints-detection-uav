# gunicorn_config.py
# Production server configuration for Render.com

bind        = "0.0.0.0:10000"   # Render uses port 10000 by default
workers     = 1                  # Keep at 1 — model is loaded in memory
threads     = 2
timeout     = 120                # Large drone images need extra time
preload_app = True               # Load model once, share across threads

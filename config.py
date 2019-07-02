import multiprocessing
from os import environ as env

PORT = int(env.get("PORT", 8080))
DEBUG_MODE = int(env.get("DEBUG_MODE", 1))

# Gunicorn config
bind = ":" + str(PORT)
workers = 2 * multiprocessing.cpu_count() + 1
threads = workers

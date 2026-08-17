"""
wsgi.py
-------
Production entry point. Gunicorn / most PaaS providers look for a module-level
`app` object, e.g.:

    gunicorn --chdir app wsgi:app --bind 0.0.0.0:8000
"""
from app import app

if __name__ == "__main__":
    app.run()

Static assets (images, js, css) are expected under /static/ served by Django. We reference:
- /static/images/* (copied from the old frontend/images)
- /static/src/styles.css and /static/src/scripts.js

In production, configure a proper STATIC_ROOT and collectstatic.
"""
Mini servidor HTTP local para servir el monitor de entrenamiento.
Ejecutar desde la carpeta del proyecto.
"""
import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 8787
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        pass  # Silenciar logs de requests

class SilentTCPServer(socketserver.TCPServer):
    """TCPServer que ignora errores de conexion abortada (WinError 10053)."""
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        import traceback
        exc = traceback.format_exc()
        # Solo ignorar errores de conexion abortada por el cliente (completamente normales)
        if "ConnectionAbortedError" in exc or "ConnectionResetError" in exc or "BrokenPipeError" in exc:
            return
        # Cualquier otro error si mostrarlo
        print(f"[Monitor] Error inesperado desde {client_address}:")
        print(exc)

def open_browser():
    import time
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/monitor.html")

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    print(f"[Monitor] Servidor iniciado en http://localhost:{PORT}/monitor.html")
    print(f"[Monitor] Sirviendo archivos desde: {DIRECTORY}")
    print(f"[Monitor] Presiona Ctrl+C para detener el servidor.\n")

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    with SilentTCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Monitor] Servidor detenido.")

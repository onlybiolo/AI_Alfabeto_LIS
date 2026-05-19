# server che riceve il flusso video dalla webcam e lo esporta


# lo importo come alias

import livecamera_exporter as lc

from flask import Flask, render_template, Response, jsonify, send_from_directory
import time
import os
import signal

server = Flask(__name__)

@server.route('/') # base che avvia index.html all'apertura del server
def index():
    return render_template("index.html") # pagina che ospita la cam

@server.route('/video_feed') # porta che mostra la webcam tramite flask
def video_feed():
    return Response(lc.TrasportatoreFrame(), mimetype="multipart/x-mixed-replace; boundary=frame")

# portiamo al client i dati (lettera, confidence, tempo)
@server.route('/get_values')
def get_values():
    lettera, confidenza, timestamp = lc.Get_Values()
    return jsonify({
        "letter": lettera,
        "confidence": confidenza,
        "time": timestamp
    })

# Rotta per mostrare la pagina di visualizzazione dei grafici e della rete
@server.route('/visualizza')
def visualizza():
    return render_template('visualizza.html')

# Rotta per permettere a visualizza.html di scaricare storico.json senza blocchi CORS
@server.route('/modelli/<path:filename>')
def serve_modelli(filename):
    return send_from_directory('modelli', filename)

@server.route('/shutdown')
def shutdown():
    # mandiamo un segnale di terminazione al processo corrente
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server in spegnimento...'

if __name__ == "__main__":
    server.run(debug=True, use_reloader=False)

// all avvio accendo lo script

let last_letter = "";
let last_time = 0;        // timestamp Unix in SECONDI dal server (quando la lettera e apparsa per la prima volta)
let last_confidence = 0;

let last_written_timestamp = 0; // timestamp dell'ultima lettera che abbiamo GIA scritto (evita duplicati)
let last_seen_ms = Date.now();  // l'ultima volta che abbiamo visto una mano (in ms locali)

let text_holder = document.getElementById("text_holder")


function GetData() {
    fetch('/get_values')
        .then(response => response.json())
        .then(data => {
            last_letter = data.letter;
            last_time = data.time;       // secondi (Python time.time())
            last_confidence = data.confidence;

            // se c'e una mano aggiorniamo quando l'abbiamo vista l'ultima volta
            if (last_letter !== "" && last_letter !== "??") {
                last_seen_ms = Date.now();
            }
        })
}

setInterval(GetData, 500)

// se una lettera viene predetta per piu di 3 secondi, la scrivo in pagina UNA SOLA VOLTA
// (last_written_timestamp tiene traccia di quale "sessione" abbiamo gia scritto)
function Scrittore() {
    const now_ms = Date.now();
    const letter_start_ms = last_time * 1000; // converto secondi → millisecondi

    const lettera_valida = last_letter !== "" && last_letter !== "??";
    const tenuta_abbastanza = (now_ms - letter_start_ms) > 1000;
    const non_ancora_scritta = last_written_timestamp !== last_time;

    if (lettera_valida && tenuta_abbastanza && non_ancora_scritta) {
        text_holder.textContent += last_letter;
        last_written_timestamp = last_time; // segniamo come gia scritta
    }

    // se non vediamo una mano da 10 secondi, resettiamo il testo
    if ((now_ms - last_seen_ms) > 10000) {
        text_holder.textContent = "";
        last_written_timestamp = 0;
    }
}

setInterval(Scrittore, 500)

function GetTime() {
    return Math.round((Date.now() - last_seen_ms) / 1000); // secondi da quando abbiamo visto la mano
}

function RIscriviTimer() {
    const secondi = GetTime();
    document.getElementById("timer").textContent = "Nessuna mano da: " + secondi + "s (reset a 10s)";
}

setInterval(RIscriviTimer, 100)

function stop_server() {
    fetch('/shutdown')
        .then(response => response.text())
        .then(data => {
            console.log(data);
            // il server si e spento, chiudiamo la scheda
            window.close();
        });
}
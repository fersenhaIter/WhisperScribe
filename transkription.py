import os
from pydub import AudioSegment
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from spectralcluster import SpectralClusterer
import numpy as np
from colorama import Fore, Style, init

# Colorama initialisieren
init(autoreset=True)

# Pfad zum aktuellen Verzeichnis
pfad = os.getcwd()

# Unterstützte Audioformate
audio_extensions = (".mp3", ".flac")

# Whisper-Modell laden
print(Fore.CYAN + "Lade Whisper-Modell...")
modell = whisper.load_model("base")

# VoiceEncoder initialisieren
print(Fore.CYAN + "Initialisiere Voice Encoder...")
encoder = VoiceEncoder()

# Alle Dateien im Verzeichnis durchgehen
for datei in os.listdir(pfad):
    if datei.endswith(audio_extensions):
        print(Fore.GREEN + f"\nVerarbeite Datei: {datei}")

        # Benutzer nach der Anzahl der Sprecher für diese Datei fragen
        print(Fore.YELLOW + f"Geben Sie die Anzahl der Sprecher für '{datei}' an.")
        print("Falls unbekannt, geben Sie einen Bereich an (z.B. '2-5').")
        speaker_input = input("Anzahl der Sprecher oder Bereich: ")

        # Verarbeiten der Benutzereingabe
        if '-' in speaker_input:
            try:
                min_speakers, max_speakers = map(int, speaker_input.split('-'))
            except ValueError:
                print(Fore.RED + "Ungültige Eingabe. Bitte einen gültigen Bereich eingeben (z.B. '2-5').")
                continue
        else:
            try:
                num_speakers = int(speaker_input)
                min_speakers = max_speakers = num_speakers
            except ValueError:
                print(Fore.RED + "Ungültige Eingabe. Bitte eine Zahl eingeben oder einen Bereich (z.B. '2-5').")
                continue

        # SpectralClusterer initialisieren
        print(Fore.CYAN + "Initialisiere Spectral Clusterer...")
        clusterer = SpectralClusterer(
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )

        # Audio konvertieren
        print(Fore.BLUE + "Konvertiere Audio...")
        audio = AudioSegment.from_file(datei)
        wav_datei = datei.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_datei, format="wav")

        # Transkription durchführen
        print(Fore.BLUE + "Führe Transkription durch...")
        ergebnis = modell.transcribe(wav_datei, language='de', task='transcribe', verbose=False)

        # Audio laden
        print(Fore.BLUE + "Lade Audio für Sprechererkennung...")
        wav_audio = preprocess_wav(wav_datei)

        # Zeitstempel der Transkription
        segments = ergebnis['segments']

        # Für jedes Segment den mittleren Audioausschnitt extrahieren
        print(Fore.BLUE + "Extrahiere Embeddings für jedes Segment...")
        embeddings = []
        valid_indices = []
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            middle_time = start_time + duration / 2

            # Extrahiere 0.5 Sekunden um die Mitte des Segments
            start_extract = max(0, middle_time - 0.25)
            end_extract = min(len(wav_audio) / 16000, middle_time + 0.25)

            audio_segment = wav_audio[int(start_extract * 16000):int(end_extract * 16000)]
            if len(audio_segment) > 0:
                embedding = encoder.embed_utterance(audio_segment)
                if np.linalg.norm(embedding) > 0:
                    embeddings.append(embedding)
                    valid_indices.append(i)
                else:
                    # Überspringen, falls Embedding-Norm Null ist
                    pass
            else:
                # Überspringen, falls kein gültiges Audio vorhanden ist
                pass

        # Clustering durchführen
        print(Fore.BLUE + "Führe Sprecher-Clustering durch...")
        if len(embeddings) > 0:
            embeddings_array = np.vstack(embeddings)
            clusters = clusterer.predict(embeddings_array)
        else:
            clusters = []

        # Transkript mit Sprecherzuordnung erstellen
        print(Fore.BLUE + "Erstelle Transkript mit Sprecherzuordnung...")
        final_text = ""
        cluster_index = 0
        for i, segment in enumerate(segments):
            if cluster_index < len(valid_indices) and i == valid_indices[cluster_index]:
                speaker = f"Sprecher_{clusters[cluster_index]}"
                cluster_index += 1
            else:
                speaker = "Unbekannt"
            text = segment['text'].strip()
            final_text += f"{speaker}: {text}\n"

        # Transkript in TXT-Datei speichern
        txt_datei = datei.rsplit(".", 1)[0] + ".txt"
        with open(txt_datei, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(Fore.GREEN + f"Transkript gespeichert als {txt_datei}")

        # Die temporäre WAV-Datei löschen
        print(Fore.YELLOW + f"Lösche temporäre Datei: {wav_datei}")
        os.remove(wav_datei)

print(Fore.MAGENTA + "\nAlle Dateien wurden verarbeitet.")


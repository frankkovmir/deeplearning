# Allgemein

Benzinapp für das Gruppenprojekt in PKI.
Teilnehmer: Sam Barjesteh, Hicham Ben Ayoub, Frank Kovmir, Sven Simon Szczesny

Das Programm kann entweder über die FuelGuru.exe im Archiv ausgeführt, oder über einen Python Interpreter
mittels Kompilierung der main.py im Hauptordner gestartet werden. (Entweder durch Import des gesamten Ordners in eine IDE, oder per Shell-Aufruf).
In beiden Fällen ist es notwendig, dass die folgende Ordnerstruktur genau so eingehalten wird.

# Ordnerstruktur
```
Benzinapp
│   .gitignore
│   main.py
│   README.md
│   Tooltip.py
│
├───.idea
│   │   .gitignore
│   │   Benzinapp.iml
│   │   misc.xml
│   │   modules.xml
│   │   vcs.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│           Project_Default.xml
│
├───data
│       historical_data.csv
│       Liste-der-PLZ-in-Excel-Karte-Deutschland-Postleitzahlen.xlsx
│
├───docs
├───environment
│       conda.yaml
│       requirements.txt
│
├───icon
│       bg.jpg
│       gasstation_4334.ico
│       graph.jpg
│       musik_bg.jpg
│       pdf_hintergrund_bild.jfif
│       speaker-2488096_1280.png
│
├───sounds
│       background.mp3
│       Control.mp3
│       Shutdown.mp3
│       Windows.mp3
│
└───src
    ├───dashboard
    │   │   main.py
    │   │   __init__.py
    │   │
    │   ├───components
    │   │       bundeslaender_dropdown.py
    │   │       fuel_dropdown.py
    │   │       ids.py
    │   │       layout.py
    │   │       line_chart.py
    │   │       year_dropdown.py
    │   │       __init__.py
    │   │
    │   └───data
    │           loader.py
    │           __init__.py
    │
    └───historical_data
            historical_data.py
            README.md
            __init__.py
```

# Environment Setup

## Installation mit Anaconda/Miniconda

Folgende Befehle vom Stammverzeichnis des Projekts ausführen.

```shell
conda env create -f ./environment/conda.yaml
conda activate benzinapp
```

## Installation mit Pip

Folgenden Befehl vom Stammverzeichnis des Projekts ausführen.

```shell
python -m pip install -r ./environment/requirements.txt
```

# Ausführen per Terminal

1.) Archiv in einen Ordner am gewünschten Ort entpacken

![grafik](https://user-images.githubusercontent.com/114833933/210900512-57357386-4a54-43f0-8600-5b5787a622d5.png)

  2.) Start des Terminals aus dem Hauptordner des entpackten Archivs (cmd im Pfad des Fensters eingeben), oder Navigation in der Shell zu dem Ordner per cd

![grafik](https://user-images.githubusercontent.com/114833933/210898763-18ee6d49-f694-4f78-bd5a-357f041bf93b.png)


3.) Ausführen des Befehls python main.py

![grafik](https://user-images.githubusercontent.com/114833933/210898866-850e875c-3b5a-4988-a616-73f1e6f24414.png)

# Ausführen per IDE (am Beispiel Pycharm)


  1.) Archiv entpacken

![grafik](https://user-images.githubusercontent.com/114833933/210900512-57357386-4a54-43f0-8600-5b5787a622d5.png)

2.) File -> Open -> Navigation zum Ordner und Öffnen

# Ausführen per Anwendung (.exe)

  1.) Archiv entpacken

2.) Die FuelGuru Anwendung per Doppelklick, oder Rechtsklick -> Ausführen starten

![grafik](https://user-images.githubusercontent.com/114833933/210901242-2117d1b2-cf99-46f8-bd47-2839cc0f339d.png)


# Troubleshoot

Das Programm ist unter Windows entwickelt und getestet worden.
Sollten bei dem Aufruf der main.py importierte Module fehlen, so sind diese per pip install 'modulname' in die genutzte Umgebung zu installieren.

![grafik](https://user-images.githubusercontent.com/114833933/210899951-b74d4360-1dee-463e-b6f9-506df495473d.png)

Sollten bei dem Start der Anwendung (.exe) Fehler entstehen, dann befindet sich die .exe vermutlich nicht im Hauptordner zusammen mit den Abhängigkeiten "icon", "sounds" und "data"

![grafik](https://user-images.githubusercontent.com/114833933/210901385-532ac98a-bffd-410a-a063-93a0af3fc61a.png)

Ein bekanntes Problem ist, dass das Dashboard für die historischen Daten bei der .exe nicht korrekt funktioniert. Um das Problem zu lösen müssen zuvor die Module "dash" und "dash-bootstrap-components" (zB per pip install) installiert werden. Ggf. muss die sich öffnende Browserseite aktualisiert werden.

Ebenso ist bei der .exe die lange Ladezeit bis zur Öffnung des Programms ein bekanntes, aber nicht ohne weiteres lösbares Problem, da viele Pakete geladen werden müssen. Hier muss man geduldig sein.

Teilweise kommt es zu Problemen bei dem für Sounds genutzten Pygame Modul in Kombination mit AnaConda (o.ä. Environments) aufgrund von einer fehlenden .dll Datei (libmpg123-0.dll). Diese sollten nur entstehen, wenn die main.py per IDE oder Terminal gestartet wird, nicht jedoch bei der .exe Datei.
Ein möglicher Workaround ist der manuelle Download einer passenden .whl (wheel) Datei.

Die Installation sollte wie folgt erfolgen:

  1.) Aufruf https://www.lfd.uci.edu/~gohlke/pythonlibs/

2.) STRG+F und Suche nach "Pygame" (ohne Anführungsstriche)

![grafik](https://user-images.githubusercontent.com/114833933/210898649-ac85cb73-1968-44ff-8a67-8ff14a8b07f3.png)

3.) Ausführen des Terminals (CMD)

4.) Auführen des Befehls "pip debug --verbose" (ohne Anführungsstriche)

![grafik](https://user-images.githubusercontent.com/114833933/210898493-277714d3-47cf-4404-99a6-54396ed6492a.png)

5.) Download einer unterstützen Pygame .whl Datei (siehe Sektion "Compatible tags" im Terminal)

6.) Start des Terminals aus dem Download-Ordner der Datei, oder Navigation zum Download-Ordner per cd

7.) Auführen des Befehls pip install "vollständiger heruntergeladener Dateiname"

![grafik](https://user-images.githubusercontent.com/114833933/210898983-b4b586c3-f87f-4f18-8854-58289eaa00e6.png)

8.) Nach Erfolgreicher Installation Neustart des Rechners und der IDE


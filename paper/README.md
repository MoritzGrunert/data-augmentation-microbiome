# Vorlage Abschlussarbeiten

Dieses Layout kann soll für Bachelor- und Masterarbeiten verwendet werden. Die
Vorlage kann für englische und deutsche Abschlussarbeiten verwendet werden.

In `thesis_meta.tex` können persönliche Informationen für das Layout definiert
werden.

Die Datei `master.tex` heißt "master", da es sich hierbei um die Masterdatei der
Abschlussarbeit handelt. Es hat nichts mit dem angestrebten Abschluss zu tun und
sollte nicht verändert werden.

## Anforderungen

* aktuelle LaTeX-Installation (alternativ kann [unser Docker-Image verwendet
  werden](https://gitlab.cs.uni-duesseldorf.de/cn-tsn/general/templates/latex/container_registry))
  * Unter Linux bist du sicher, wenn du das Paket `texlive-full` installierst.
* [git-lfs](https://git-lfs.github.com/) (sorgt dafür, dass in der
  [.gitattributes](.gitattributes) gelistete Dateitypen ins LFS geschoben
  werden)

## Benutzung/Compilieren

Eigene LaTeX-Dateien müssen an der markierten Stelle in der `master.tex` Datei
eingebunden werden.

### mit latexmk

`latexmk` kann verwendet werden, um die Arbeit zu bauen (konfiguriert in `.latexmkrc`). Der Befehl dafür sieht wie
folgt aus:

    latexmk

Wenn latexmk die PDF automatisch bauen soll, wenn sich eine der eingebundenen
Dateien verändert, so sieht der Aufruf so aus:

    latexmk -pvc

Das funktioniert analog auch mit unserem LaTeX-Image:

    docker run --rm -v `pwd`:/tex gitlab.cs.uni-duesseldorf.de:5001/cn-tsn/general/templates/latex:latest latexmk

oder mit automatischem Build, wenn eine `.tex`-Datei sich ändert:

    docker run --rm -v `pwd`:/tex gitlab.cs.uni-duesseldorf.de:5001/cn-tsn/general/templates/latex:latest latexmk -pvc

### manuell auf der Konsole

Beachte, dass mindestens initial `pdflatex` mehrfach ausgeführt werden muss, damit Referenzen richtig erzeugt werden.

    pdflatex --shell-escape master.tex # 1. Durchlauf, schreibt Referenzen (Seitenzahlen, Zitationen, Abbildungsnummern, …) in Hilfsdatei
    biblatex master # erzeugt Literaturangaben/Referenzbezeichnungen für zitierte Literatur
    pdflatex --shell-escape master.tex # 2. Durchlauf, kompiliert mit korrekten Referenzen
    pdflatex --shell-escape master.tex # 3. Durchlauf, da sich durch eingefügte Referenzen ggf. Seitenzahlen verschoben haben

### LaTeX-Editor

Alternativ kann man die master-Datei auch automatisch von eigenen LaTeX-Editor
bauen lassen.

### Sprache ändern

Die Sprache kann in der Datei `master.tex` geändert werden. Einfach die
entsprechenden Zeilen auskommentieren / wechseln und schon ändert sich das
Layout.

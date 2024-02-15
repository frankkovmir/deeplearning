"""
author: Frank Kovmir
frankkovmir@gmail.com
"""

import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Konstruktor der Netzwerk-Klasse.

        Parameter:
        - input_size: Die Größe des Eingabevektors, was der ANzahl der Merkmale entspricht, die  das Netzwerk erhält (5, siehe get_state methode in main.py).
        - output_size: Die Größe des Ausgabevektors, was der Anzahl der Aktionen entspricht, die das Netzwerk treffen kann (zwei, links/rechts).

        Das Netzwerk besteht aus drei vollständig verbundenen Schichten (fully connected layers).
        """

        super(DeepQNetwork, self).__init__()


        # Definieren der ersten vollständig verbundenen Schicht (fc1)
        # von der Eingabeschicht zur ersten verborgenen Schicht mit 128 Neuronen
        self.fc1 = nn.Linear(input_size, 128)

        # Definieren der zweiten vollständig verbundenen Schicht (fc2)
        # von der ersten zur zweiten verborgenen Schicht, ebenfalls mit 128 Neuronen
        self.fc2 = nn.Linear(128, 128)

        # Definieren der dritten vollständig verbundenen Schicht (fc3)
        # von der letzten verborgenen Schicht zur Ausgabeschicht
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        """
        Definiert den Forward-Pass des Netzwerks.
        Diese Methode wird aufgerufen, um die Ausgabe des Netzwerks basierend auf den Eingabedaten zu berechnen.

        Parameter:
        - x: Eingabedaten, ein Tensor, der in das Netzwerk gefüttert wird.

        Die Aktivierungsfunktion ReLU (Rectified Linear Unit) wird auf die ersten beiden Schichten angewendet.
        """

        # Anwenden der ReLU-Aktivierungsfunktion nach der ersten Schicht
        x = F.relu(self.fc1(x))

        # Anwenden der ReLU-Aktivierungsfunktion nach der zweiten Schicht
        x = F.relu(self.fc2(x))

        # Rückgabe der Ausgabe der dritten Schicht ohne zusätzliche Aktivierungsfunktion
        return self.fc3(x)
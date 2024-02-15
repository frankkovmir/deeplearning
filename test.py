import argparse
import torch
from main import SpaceDodgerGame
import pygame
from model import DeepQNetwork
import os

def get_args():
    """
    Definiert und holt die Befehlszeilenargumente für das Testskript.
    --saved_path: Pfad zum Verzeichnis, in dem das trainierte Modell gespeichert ist.
    --num_episodes: Anzahl der Episoden, über die das Modell getestet wird.
    """
    parser = argparse.ArgumentParser("""Testet das Deep Q Network im SpaceDodger-Spiel""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_episodes", type=int, default=10)  # Standardmäßig 10 Episoden
    args = parser.parse_args()
    return args

def test(opt):
    """
    Testet das trainierte Deep Q-Network-Modell im SpaceDodger-Spiel mit der definierten ANzahl Episoden
    """
    input_size = 5  # Größe des Eingabevektors
    output_size = 2  # Größe des Ausgabevektors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork(input_size, output_size).to(device)

    model_path = f"{opt.saved_path}/space_dodger_final.pth"
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        print("Modellgewichte erfolgreich geladen.")
    else:
        print("Kein gespeichertes Modell gefunden. Bitte überprüfen Sie den Pfad.")
        return

    model.eval()  # Modell in den Evaluierungsmodus setzen

    for episode in range(opt.num_episodes):  # Schleife über die gewünschte Anzahl an Episoden
        game = SpaceDodgerGame()  # Erstellt eine neue Instanz des Spiels
        state = game.reset()  # Setzt das Spiel zurück und erhält den Anfangszustand

        while True:
            state_tensor = torch.tensor([state], dtype=torch.float).to(device)
            with torch.no_grad():
                prediction = model(state_tensor)
            action = torch.argmax(prediction).item()

            next_state, _, done = game.step(action)  # Führt die gewählte Aktion im Spiel aus

            state = next_state

            pygame.display.flip()  # Aktualisiert das Spiel-Display
            game.render(action)  # Zeichnet das Spiel mit der gewählten Aktion

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if done:  # Überprüft, ob die Episode beendet ist
                break

        print(f"Episode {episode + 1} abgeschlossen.")

if __name__ == "__main__":
    opt = get_args()
    test(opt)

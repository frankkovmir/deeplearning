"""
author: Frank Kovmir
frankkovmir@gmail.com
"""

import pygame
import random

class SpaceDodgerGame:
    def __init__(self):
        """
        Initialisiert das Spiel und setzt grundlegende Parameter wie Spielgeschwindigkeit und Grafikelemente.
        """

        pygame.init()

        self.clock = pygame.time.Clock()
        self.dodge_reward = 0.1  # Belohnungspunkte für das Ausweichen
        self.asteroids_per_level = 1  # Anzahl der Asteroiden pro Level
        self.current_level = 1  # Startlevel
        self.spawned_asteroids_count = 0  # Anzahl der bereits erschienenen Asteroiden

        # Parameter für die Lücke in der Asteroidenkette
        self.hole_position = 0
        self.hole_size = 0

        # Laden des Asteroiden-Sprite-Sheets und Extraktion einzelner Sprites
        asteroid_sheet = pygame.image.load('assets/asteroids.png')
        asteroid_sprite_width, asteroid_sprite_height = asteroid_sheet.get_size()
        asteroid_sprite_width //= 8
        asteroid_sprite_height //= 8

        self.asteroid_sprites = []
        for row in range(8):
            for col in range(8):
                rect = pygame.Rect(col * asteroid_sprite_width, row * asteroid_sprite_height, asteroid_sprite_width, asteroid_sprite_height)
                asteroid_sprite = asteroid_sheet.subsurface(rect)
                self.asteroid_sprites.append(asteroid_sprite)

        # Laden der Raumschiff-Sprites
        sprite_sheet = pygame.image.load('assets/ship.png')
        sprite_width, sprite_height = sprite_sheet.get_size()
        sprite_width //= 5
        sprite_height //= 2
        scale_factor = 2
        self.spaceships = []

        for i in range(5):
            rect = pygame.Rect(i * sprite_width, 0, sprite_width, sprite_height)
            spaceship_sprite = sprite_sheet.subsurface(rect)
            scaled_sprite = pygame.transform.scale(spaceship_sprite,
                                                   (sprite_width * scale_factor, sprite_height * scale_factor))
            self.spaceships.append(scaled_sprite)

        # Erstellen von Masken für die Kollisionserkennung
        self.spaceship_masks = [pygame.mask.from_surface(sprite) for sprite in self.spaceships]

        # Variablen für die Animation des Raumschiffs
        self.current_spaceship_index = 0
        self.animation_frame_count = 0
        self.animation_speed = 5  # Lower is faster

        # Spielfenster-Größe
        self.width, self.height = 288, 512
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Dodger")

        # Eigenschaften des Raumschiffs
        self.spaceship_width, self.spaceship_height = self.spaceships[0].get_size()
        self.spaceship_speed = 3
        self.spaceship = None

        # Eigenschaften der Asteroiden
        self.asteroid_size = 40
        self.asteroid_speed = 1
        self.asteroid_spawn_rate = 100 # not spawning simultatnously
        self.asteroids = []
        self.reset()

    def reset(self):
        """
        Setzt das Spiel nach einer Kollision zurück. Das Raumschiff und die Asteroiden werden neu positioniert und das Level wird zurückgesetzt.
        """

        self.spaceship = pygame.Rect(self.width // 2 - self.spaceship_width // 2,
                                     self.height - self.spaceship_height - 10,
                                     self.spaceship_width, self.spaceship_height)
        self.asteroids = []
        self.frame_count = 0
        self.spaceship_mask = self.spaceship_masks[0]
        self.current_level = 1
        self.spawned_asteroids_count = 0
        self.asteroid_speed = 1
        return self.get_state()

    def update_spaceship_animation(self):
        """
        Aktualisiert die Animation des Raumschiffs durch Wechsel der Sprites.
        """

        self.animation_frame_count += 1
        if self.animation_frame_count >= self.animation_speed:
            self.animation_frame_count = 0
            self.current_spaceship_index = (self.current_spaceship_index + 1) % len(self.spaceships)
            self.spaceship_mask = self.spaceship_masks[self.current_spaceship_index]

    def spawn_asteroid_chain(self):
        """
        Erzeugt eine Kette von Asteroiden mit einer Lücke, durch die das Raumschiff hindurchfliegen kann. Die Position und Größe der Lücke werden zufällig bestimmt.
        """

        # Festlegen des Bereichs, in dem das Loch erscheinen kann, um die Startposition des Raumschiffs zu vermeiden
        min_x = 0
        max_x = self.width - self.asteroid_size * 3

        # Bestimmung der aktuellen Position des Raumschiffs
        spaceship_x_center = self.spaceship.x + self.spaceship_width // 2

        # Definieren eines Bereichs um das Raumschiff, in dem das Loch nicht erzeugt werden sollte
        exclusion_zone = self.spaceship_width * 2

        # Aufteilen der möglichen Lochpositionen in zwei Bereiche, um die Ausschlusszone zu vermeiden
        left_range = [x for x in range(min_x, spaceship_x_center - exclusion_zone) if
                      x + self.asteroid_size * 3 <= max_x]
        right_range = [x for x in range(spaceship_x_center + exclusion_zone, max_x) if x >= min_x]

        # Überprüfen, ob gültige Positionen in den Bereichen vorhanden sind
        if not left_range and not right_range:
            print("No valid positions for the hole. Adjust game parameters.")
            return

        # Zufällige Auswahl, ob das Loch links oder rechts von der Ausschlusszone sein wird
        if left_range and (not right_range or random.random() > 0.5):
            hole_position = random.choice(left_range)
        else:
            hole_position = random.choice(right_range)

        # Bestimmen der Größe des Lochs
        hole_size = random.randint(self.asteroid_size * 2, self.asteroid_size * 3)

        # Erzeugen der Asteroiden, wobei eine Lücke an der gewählten Position gelassen wird
        for x in range(0, self.width, self.asteroid_size):
            if not hole_position <= x < hole_position + hole_size:
                asteroid_sprite = random.choice(self.asteroid_sprites)
                scaled_sprite = pygame.transform.scale(asteroid_sprite, (self.asteroid_size, self.asteroid_size))
                self.asteroids.append(
                    [scaled_sprite, pygame.Rect(x, 0, self.asteroid_size, self.asteroid_size), self.asteroid_speed])

        # Aktualisieren der Lochposition und -größe
        self.hole_position = hole_position
        self.hole_size = hole_size

    def step(self, action):
        """
        Führt einen Schritt im Spiel durch, basierend auf der übergebenen Aktion.
        Aktualisiert den Spielzustand und gibt den neuen Zustand, die Belohnung und ein Flag zurück, ob das Spiel beendet ist.
        """

        collision_penalty = -1
        successful_hole_navigation_reward = 0.5  # Belohnung für das erfolgreiche Navigieren durch die Lücke
        reward = 0
        done = False

        # Bewegungsaktionen des Agenten
        if action == 0 and self.spaceship.x > 0:
            self.spaceship.x -= min(self.spaceship_speed, self.spaceship.x)
        elif action == 1 and self.spaceship.x < self.width - self.spaceship_width:
            self.spaceship.x += min(self.spaceship_speed, self.width - self.spaceship_width - self.spaceship.x)

        # Erscheinen neuer Asteroidenketten
        if self.spawned_asteroids_count < self.asteroids_per_level:
            if self.frame_count % self.asteroid_spawn_rate == 0:
                self.spawn_asteroid_chain()
                self.spawned_asteroids_count += 1

        # Berechnen der Position und Größe der Lücke zur Überganbe an den Agenten
        if self.asteroids:
            all_x_positions = [asteroid[1].x for asteroid in self.asteroids]
            hole_start = max(0, min(all_x_positions) - self.asteroid_size)
            hole_end = min(self.width, max(all_x_positions) + self.asteroid_size)
        else:
            hole_start = 0
            hole_end = self.width

        # Kollision überprüfen und Belohnung oder Strafe verteilen

        for asteroid in self.asteroids:
            asteroid_sprite, asteroid_rect, _ = asteroid
            prev_y = asteroid_rect.y
            asteroid_rect.y += self.asteroid_speed
            new_y = asteroid_rect.y

            asteroid_mask = pygame.mask.from_surface(asteroid_sprite)
            offset_x = asteroid_rect.x - self.spaceship.x
            offset_y = new_y - self.spaceship.y

            if self.spaceship_mask.overlap(asteroid_mask, (offset_x, offset_y)):
                done = True
                reward = collision_penalty
                self.reset()
                break

            if prev_y < self.height and new_y >= self.height:
                if hole_start <= self.spaceship.x <= hole_end:
                    reward += successful_hole_navigation_reward

        # Asteroiden entfernen die den utnenren Rand verlassen haben
        self.asteroids = [asteroid for asteroid in self.asteroids if asteroid[1].y <= self.height]

        # Levelübergang vorbereiten
        if len(self.asteroids) == 0 and self.spawned_asteroids_count >= self.asteroids_per_level:
            self.prepare_next_level()
            done = False

        self.frame_count += 1

        # print(f'state {self.get_state()}')
        # print(f'reward {reward}')
        # print(f'done {done}')

        return self.get_state(), reward, done
    def prepare_next_level(self):
        """Übergang ins nächste Level
        """
        self.current_level += 1
        #self.asteroids_per_level += 1
        self.asteroid_speed += 0.5
        self.spawned_asteroids_count = 0
        self.asteroids = []
        print(f"Nächstes Level: {self.current_level}")

    def get_state(self):
        """
        Gibt den aktuellen Zustand des Spiels zurück. Dieser Zustand wird in Form einer Liste von Werten zurückgegeben, die verschiedene Aspekte des Spiels repräsentieren.
        Dazu gehören die X- und Y-Position des Raumschiffs, die Position und Größe der Lücke in der Asteroidenkette und die Geschwindigkeit der Asteroiden.
        """

        pygame.display.flip()

        state = [
            self.spaceship.x / self.width,  # Relative X-Position des Raumschiffs
            self.spaceship.y / self.height,  # Relative Y-Position des Raumschiffs
            self.hole_position / self.width,  # Relative X-Position der Lücke
            self.hole_size / self.width,  # Relative Größe der Lücke
            self.asteroid_speed  # Geschwindigkeit der Asteroiden
        ]
        #print(state)
        return state

    def render(self,action):
        """
        Zeichnet alle Spielelemente auf dem Bildschirm. Dies umfasst das Raumschiff, die Asteroiden und die Spielinformationen wie das aktuelle Level
        und die ausgeführte Aktion (Debugzwecke).
        """

        self.screen.fill((0, 0, 0))

        current_spaceship = self.spaceships[self.current_spaceship_index]
        self.screen.blit(current_spaceship, self.spaceship.topleft)

        for asteroid in self.asteroids:
            self.screen.blit(asteroid[0], asteroid[1].topleft)

        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level: {self.current_level}", True, (255, 255, 255))
        action_text = font.render(f"Action: {action}", True, (255, 255, 255))
        self.screen.blit(action_text, (10, 50))
        self.clock.tick(60)
        self.screen.blit(level_text, (10, 10))

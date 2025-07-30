from flask import Flask, render_template, Response
import cv2
import numpy as np
import pygame
import sys
import random
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from neural_network import NeuralNetwork, create_population, crossover

app = Flask(__name__)

# Initialize Pygame
print("Initializing Pygame...")
pygame.init()
print("Pygame initialized.")

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Game variables
GRAVITY = 0.75
BIRD_JUMP = -12

# Colors
WHITE = (255, 255, 255)

# Initialize the screen
print("Setting up Pygame screen...")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
print("Pygame screen setup complete.")

# Create a plain white background surface
bg_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
bg_surface.fill(WHITE)

# Genetic Algorithm Parameters
population_size = 50
mutation_rate = 0.2
elite_size = 5
generation = 0
best_score = 0

# Create the initial population
print("Creating initial population...")
population = create_population(population_size, 5, 20, 1)
print("Initial population created.")

# Bird and movement setup
initial_bird_rects = [pygame.Rect(100, SCREEN_HEIGHT // 2, 30, 30) for _ in range(population_size)]
bird_movements = [0 for _ in range(population_size)]

# Pipe setup
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)
pipe_height = [300, 400, 500]

def draw_bird(bird_rect, color):
    pygame.draw.ellipse(screen, color, bird_rect)

def draw_pipes(pipes):
    for pipe in pipes:
        pygame.draw.rect(screen, (0, 0, 0), pipe)

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    return [pipe for pipe in pipes if pipe.right > 0]

def check_collision(pipes, bird_rect):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return False
    if bird_rect.top <= -100 or bird_rect.bottom >= SCREEN_HEIGHT:
        return False
    return True

def create_pipe():
    random_pipe_pos = random.choice(pipe_height)
    bottom_pipe = pygame.Rect(SCREEN_WIDTH + 100, random_pipe_pos, 50, SCREEN_HEIGHT)
    top_pipe = pygame.Rect(SCREEN_WIDTH + 100, random_pipe_pos - 150 - SCREEN_HEIGHT, 50, SCREEN_HEIGHT)
    return bottom_pipe, top_pipe

def reset_game():
    global initial_bird_rects, bird_movements, pipe_list
    initial_bird_rects = [pygame.Rect(100, SCREEN_HEIGHT // 2, 30, 30) for _ in range(population_size)]
    bird_movements = [0 for _ in range(population_size)]
    pipe_list = []

def normalize_input(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def generate_frame():
    global bird_movements, pipe_list, generation, best_score, population
    reset_game()
    current_population = population
    fitness_scores = [0] * population_size
    alive_birds = [True] * population_size

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == SPAWNPIPE:
                pipe_list.extend(create_pipe())

        screen.blit(bg_surface, (0, 0))

        # Pipes
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)

        # AI control
        if pipe_list:
            for i, bird in enumerate(current_population):
                if alive_birds[i]:
                    bird_x = initial_bird_rects[i].centerx
                    bird_y = initial_bird_rects[i].centery
                    pipe_x = pipe_list[0].centerx
                    pipe_y = pipe_list[0].centery
                    pipe_gap_top = pipe_list[0].top
                    pipe_gap_bottom = pipe_list[0].bottom

                    # Normalize inputs to keep them within a reasonable range
                    inputs = np.array([
                        normalize_input(bird_y, 0, SCREEN_HEIGHT),
                        normalize_input(bird_movements[i], -20, 20),
                        normalize_input(pipe_x - bird_x, -SCREEN_WIDTH, SCREEN_WIDTH),
                        normalize_input(pipe_gap_bottom - bird_y, -SCREEN_HEIGHT, SCREEN_HEIGHT),
                        normalize_input(pipe_gap_top - bird_y, -SCREEN_HEIGHT, SCREEN_HEIGHT)
                    ]).reshape(-1, 1)

                    output = bird.feedforward(inputs)
                    if output[0] > 0.5:
                        bird_movements[i] = BIRD_JUMP

                    # Bird
                    bird_movements[i] += GRAVITY
                    initial_bird_rects[i].centery += bird_movements[i]
                    color_intensity = int(255 * (i / population_size))
                    color = (color_intensity, color_intensity, color_intensity)
                    draw_bird(initial_bird_rects[i], color)

                    # Fitness is the distance traveled and pipes passed
                    fitness_scores[i] += 1
                    if initial_bird_rects[i].centerx > pipe_list[0].centerx:
                        fitness_scores[i] += 10

                    # Check for collision
                    if not check_collision(pipe_list, initial_bird_rects[i]):
                        alive_birds[i] = False

        # Capture the screen as an image
        image = pygame.surfarray.pixels3d(screen)
        image = np.rot90(image)  # Correct the rotation
        image = np.flipud(image)  # Correct the upside down issue
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        pygame.display.update()
        clock.tick(30)

        # Evolve the population after each generation
        if all(not alive for alive in alive_birds):
            best_index = np.argmax(fitness_scores)
            best_score = max(best_score, fitness_scores[best_index])
            best_bird = current_population[best_index]

            # Select the top N birds (elite) and retain them
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = [current_population[i] for i in elite_indices]

            # Create the new population with crossover and mutation
            new_population = elite[:]
            while len(new_population) < population_size:
                parent1, parent2 = random.choices(elite, k=2)
                child = crossover(parent1, parent2)
                child.mutate(mutation_rate)
                new_population.append(child)

            current_population = new_population
            fitness_scores = [0] * population_size
            alive_birds = [True] * population_size
            generation += 1
            reset_game()

def visualize_neural_network(nn):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.9, 'Neural Network', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    for i in range(nn.input_size):
        ax.text(0.1, 0.8 - i * 0.2, f'Input {i+1}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    for i in range(nn.hidden_size):
        ax.text(0.5, 0.8 - i * 0.2, f'Hidden {i+1}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    for i in range(nn.output_size):
        ax.text(0.9, 0.8 - i * 0.2, f'Output {i+1}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close(fig)
    return img_base64

# Generate the initial neural network visualization before starting the server
print("Generating initial neural network visualization...")
nn_image = visualize_neural_network(population[0])
print("Initial neural network visualization generated.")

@app.route('/')
def index():
    return render_template('index.html', nn_image=nn_image)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jump', methods=['POST'])
def jump():
    global bird_movement
    bird_movement = BIRD_JUMP
    return '', 204

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
    print("Flask server started.")

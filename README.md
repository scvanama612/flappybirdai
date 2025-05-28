# Flappy Bird AI



https://github.com/Ori2846/FlappyBirdAI/assets/74078771/1e63f661-defa-4936-8d23-da5539d5149b



This project implements a genetic algorithm and neural networks to train AI agents to play Flappy Bird. The AI birds evolve over generations to improve their performance in the game.

## Overview

The AI agents (birds) are controlled by a neural network. The neural network takes inputs about the bird's environment and outputs a decision whether the bird should jump. The goal is to navigate through the pipes without colliding. The fitness of each bird is determined by how far it travels and how many pipes it passes.

## Features

- **Neural Network**: The birds are controlled by a neural network with a customizable number of input, hidden, and output neurons.
- **Genetic Algorithm**: The neural networks are trained using a genetic algorithm, which includes crossover and mutation to evolve the birds over generations.
- **Real-time Visualization**: The game can be visualized in real-time to observe the birds' performance.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- Pygame
- OpenCV
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ori2846/FlappyBirdAI.git
    cd FlappyBirdAI
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

### Controls

- Press the `Space` key to make the bird jump.

## Code Structure

- `app.py`: Main Flask application file that runs the server and handles the game loop.
- `neural_network.py`: Contains the implementation of the neural network and genetic algorithm functions.
- `templates/index.html`: HTML template for the web interface.

## How It Works

1. **Initialization**: The game initializes a population of neural networks (birds) with random weights.
2. **Game Loop**: Each bird plays the game using its neural network to decide when to jump based on inputs such as the bird's position and the position of the nearest pipes.
3. **Fitness Calculation**: Each bird's fitness is calculated based on how far it travels and how many pipes it passes.
4. **Selection and Reproduction**: The top-performing birds are selected to create a new generation through crossover and mutation.
5. **Evolution**: The process repeats for multiple generations, gradually improving the birds' performance.

## Future Improvements

- Implement more complex neural network architectures.
- Add additional game elements and obstacles.
- Improve the visualization and user interface.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Inspired by the original Flappy Bird game.
- Uses Pygame for game rendering and control.

## About the Maintainer

This project is actively maintained by Sai Chandra Vanama. Sai is a Software Developer with over 3 years of professional experience, specializing in Python, SQL, R, Pandas, NumPy, and Seaborn.

For inquiries or collaborations, you can reach Sai at:
-   **Email**: chandravanama1149@gmail.com
-   **GitHub**: Sai Chandra Vanama
-   **LinkedIn**: Sai Chandra Vanama
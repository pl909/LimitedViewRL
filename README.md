# Pursuit-Evasion: A PyBullet-based Multi-Agent Reinforcement Learning Environment

This project implements a pursuit-evasion environment using PyBullet physics engine, designed for multi-agent reinforcement learning research.  Currently, it focuses on a single quadrotor pursuing a stationary target.

## Features

* **PyBullet Physics Engine:** Realistic simulation of a quadrotor's dynamics.
* **Gym Environment:**  Integrated with the OpenAI Gym framework for easy integration with reinforcement learning algorithms.
* **Customizable Action and Observation Spaces:** Allows for flexibility in designing control and observation strategies.
* **Reward Function:**  Encourages the quadrotor to reach a target position while penalizing excessive distance from the target or exceeding a time limit.
* **Training and Showcase Scripts:** Provided scripts demonstrate how to train a DDPG agent and showcase its performance.
* **URDF Quadrotor Model:** A URDF file defines the quadrotor's physical properties and visual representation.

## Usage

The environment can be used with any OpenAI Gym compatible reinforcement learning algorithm. The provided scripts demonstrate using Stable Baselines3's DDPG algorithm.

To run a showcase of a pre-trained model:

```bash
python showcase_model.py
```

To train a new model:

```bash
python training_model.py
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd Pursuit-Evasion
    ```
3.  Install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Technologies Used

* **Python:** The primary programming language for the entire project.
* **OpenAI Gym:**  Framework for developing and comparing reinforcement learning algorithms.
* **PyBullet:** Physics engine used for realistic simulation of the quadrotor.
* **NumPy:**  Provides support for numerical operations and array manipulation.
* **Matplotlib:** Used for plotting and visualization (although not directly used in the core environment, it is a dependency).
* **Control:** Python control systems library (used for LQR controller calculations in `utils.py`).
* **Stable Baselines3:**  A reinforcement learning library used for training the DDPG agent.

## Configuration

The environment's parameters can be adjusted within the `PursuitEvasionEnv` class in `pursuit_evasion/envs/pursuit_evasion_env.py`.  For example, the `trainingMode` flag controls whether the PyBullet server runs in GUI or DIRECT mode.


## Dependencies

The project dependencies are listed in `requirements.txt` and `setup.py`.  They can be installed using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## Testing

The `check_env.py` and `doublecheck_env.py` files contain simple tests for basic Gym environment functionality and a basic gym loop, respectively. More comprehensive tests should be added.




*README.md was made with [Etchr](https://etchr.dev)*
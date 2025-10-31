# Comparing model based and model free reinforcement learning
This project implements model based and model free 
reinforcement learning algorithms and compares their performance
in order to evaluate model based reinforcement learning overall.

## Setup
To run the project, run `pip install requirements.txt` to install dependancies.  
Then run `main.py`.

## Modification
The project is designed to easy to modify and add new algorithms.
To add a new algorithm, create a class for it, `RandomAction.py` works well as a template.  Then add the new class to
`algorithm_list` in main.py.
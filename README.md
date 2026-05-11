## Quantum Markov Models
A collection of files necessary for understanding and testing Classical-Quantum Markov Chains.

### Whats inside
- compute_unitary.ipynb
- compute_unitary.py
- setup_qmm.sh
- README.md

### Concept
- Python and Jupyter notebook files:<br>
  Changes in the .py file are reflected into the .ipynb file by syncing (automatically).
  This serves both writing a Python code file, ready to be executed in the console and having seperate, individual cells ready to be executed in a Jupyter notebook.
- The configuration setup includes syncing the files, running the Jupyter server, running Qiskit library commands and others.

### First time setup
- Virtual Environment Setup:<br>
  Run executable to create a Python virtual environment and download all necessary libraries. Deleting the virtual environment removes modules and frees space.
- Alias Setup:<br>
  Setup an alias to quickly activate your browser Jupyter lab. Port-forward in **8888**.
  ```
  alias jupstart='cd ~; source ~/jupyter-venv/bin/activate; jupyter lab --no-browser --port=8888'
  ```

### Quick start
- Activate and hop in:
  ```
  $ jupstart
  ```
- Access in broswer:
  ```
  localhost:8888
  ```

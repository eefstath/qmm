#!/bin/bash

# Date:		11/5/2026

readonly VENV_NAME="jupyter-venv"			# Venv strict name

echo "> V ENV ${VENV_NAME} creating..."

pushd ~/						# Go to home dir

python3 -m venv ${VENV_NAME} --without-pip		# Readonly envs (SD)
source ${VENV_NAME}/bin/activate			# Activate

echo "> ${VENV_NAME} activated..."

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py	# Readonly envs
python3 get-pip.py
rm get-pip.py

pip install --upgrade pip				# Install modules
pip install numpy matplotlib networkx \
            qiskit qiskit-aer qiskit-ibm-runtime \
            jupytext jupyterlab ipykernel python-dotenv

echo "> Modules installed..."

python3 -m ipykernel install --user \
	--name=${VENV_NAME} \
	--display-name "QMM Virtual" 			# Jup Kernel register

echo "> Kernel registered..."

if [ -f "computer.unitary.py" ]; then 			# Pair files
	echo "> Pairing python to .ipyng"
	jupytext --set-formats ipynb,py:percent compute_unitary.py
fi

echo "> Sync py, ipynb files ON..."

popd							# Go where you were

echo "> V ENV ${VENV_NAME} completed..."
echo "> Use "jupstart" next time to hop in."

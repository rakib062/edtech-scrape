#Install pyenv for version management

# Install pyenv
curl https://pyenv.run | bash

# Follow the instruction to modify ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
exec "$SHELL"


# Install the latest Python from source code
pyenv install 3.9.10

# Check installed Python versions
pyenv versions

# Switch Python version
pyenv global 3.9.10

# Check where Python is actually installed
pyenv prefix
# Check the current Python version
python -V


#Add virtual environemnt as jupyter kernel
pip install jupyterlab
pip install --user ipykernel
python -m ipykernel install --user --name=[myenv]
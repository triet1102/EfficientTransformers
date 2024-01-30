#!/bin/bash
set -x
set -e

apt-get update
# prepare ubuntu for building python
apt-get install -y build-essential libz-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev lzma xz-utils liblzma-dev
apt-get install htop
# ... setup git ssh keys
# setup git config here
git config --global user.email "t.triet.1102@gmail.com"
git config --global user.name "triet1102"
git config --global credential.helper store

# install pyenv
curl https://pyenv.run | bash
# init pyenv in .bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
# init pyenv in .profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

# install direnv
curl -sfL https://direnv.net/install.sh | bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc

# restart the shell
exec "$SHELL"

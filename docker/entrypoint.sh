#!/bin/bash
# Author : Jaewon Lee (https://github.com/Lee-JaeWon)
 
set -e

echo "alias cw='cd ~/workspace'" >> ~/.bashrc
echo "alias cs='cd ~/workspace/src'" >> ~/.bashrc
echo "alias sb='source ~/.bashrc'" >> ~/.bashrc

cd /root/workspace

echo "============== mix Docker Env Ready================"

exec "$@"

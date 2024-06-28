#!/bin/bash

# Start Xvfb on display :99 with a virtual screen size
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99

# Run the command specified as arguments to the script
exec "$@"

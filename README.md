# Project Mycelium backend
frontend -> https://github.com/KoenMW/project-mycelium-fe

# Set up

Set Up a Virtual Environment by using Anaconda

Create a virtual environment in shell
- conda create --name mdenv python=3.11

Activate using
- conda activate mdenv

Install Flask within the virtual environment:
- pip install Flask

# Run the app

flask --app app run --debug

# Background

This is the backend of an application that can recognize growth in mycelium buckets from DDSS. This is done by computer vision models designed for this task. It is done by clustering and classifiyng the images in days.

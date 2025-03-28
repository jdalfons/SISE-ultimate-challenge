# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Create a new user with a specific UID and switch to it
RUN useradd -m -u 1000 user
# Set the PATH environment variable
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container with the correct ownership
COPY --chown=user ./requirements.txt requirements.txt

# Switch to root user to install necessary build tools and dependencies
USER root
RUN apt-get update && apt-get install -y gcc portaudio19-dev && rm -rf /var/lib/apt/lists/*

# Switch back to the user
USER user

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container with the correct ownership
COPY --chown=user . /app

# Expose the port that Streamlit will use
EXPOSE 7860

# Command to run the Streamlit app on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
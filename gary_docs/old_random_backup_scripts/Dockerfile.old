# Use an official Python runtime as a parent image with CUDA support
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed system dependencies (example: ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /usr/src/app
COPY requirements.txt ./

# Install most Python dependencies, excluding the conflicting ones
RUN pip --default-timeout=100 install --no-cache-dir -r requirements.txt 

# Copy the rest of the application's code
COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for number of gunicorn workers
ENV WORKERS=4

# Use the entrypoint script to start the services
CMD ["./entrypoint.sh"]

# Use the official Node.js runtime as a parent image
FROM node:14

# Set the working directory in the container
WORKDIR /usr/src/express-server

# Copy package.json and package-lock.json into the container at /usr/src/express-server
COPY package*.json ./

# Install any needed packages specified in package.json
RUN npm install

# Bundle the app source inside the Docker image
COPY . .

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Define the command to run the app using CMD which defines your runtime
CMD ["node", "server.js"]

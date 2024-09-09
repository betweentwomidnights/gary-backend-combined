# gary

This repository serves as the backend for two applications: **gary-on-the-fly** and **gary4live**. 

there's now a gary_docs folder. you should go read that instead of this.



the combined backend can be run using `docker-compose up` from the terminal.

### gary4live

![gary4live](./gary4live%20screenshot.png)

**gary4live** is a max for live device that enables musicgen continuations inside ableton. there's no text prompting here. instead, think of each fine-tune as a "preset" in the VST.

#### backend for gary4live

we have servers running to host the backend, but if you're chad enough and want one all to yourself, the backend for running **gary4live** on its own is defined in the `docker-compose-g4lwebsockets.yml` file.

just rename this file to `docker-compose.yml` in order to run it. you can rename the existing `docker-compose.yml` to something else for now.

install docker and docker-compose in your environment.

The front-end repository for **gary4live** can be found [here](https://github.com/betweentwomidnights/gary4live). 

There is an installer for mac and pc. or you can build the electron UI yourself using that repository.

you'll need ableton live. you can use gary with the 30 day trial of ableton if you want.

## installation

1. **install docker and docker compose**
   - Follow the instructions on the [Docker website](https://docs.docker.com/get-docker/) to install Docker.
   - Follow the instructions on the [Docker Compose website](https://docs.docker.com/compose/install/) to install Docker Compose.

2. **Clone this repository**

   ```sh
   git clone https://github.com/betweentwomidnights/gary-backend-combined.git
   cd gary-backend-combined
   mv docker-compose-g4lwebsockets.yml docker-compose.yml
   sudo docker build -t thecollabagepatch/g4lwebsockets:latest -f Dockerfile.g4lwebsockets .
   sudo docker build -t thecollabagepatch/redis:latest -f Dockerfile.redis .
   sudo docker-compose up

### gary-on-the-fly

![gary-on-the-fly](./gotf%20screenshot.png)

this backend (`Dockerfile.concurrent_gary`) is for the browser extension known as gary-on-the-fly. it uses yt-dlp in combination with the timestamp of the user's current youtube.com/watch url to do a musicgen continuation. then, the generations can be extended/cropped and arranged in the newtab component.

the front-end for gary-on-the-fly is at (https://github.com/betweentwomidnights/gotf-frontend.git)

there's also a web app at https://thecollabagepatch.com (go to gary's page and click gary4web)
i still need to push the web app front-end to github. gary4web uses the `concurrent_gary` setup.

a third backend exists for a multi-gpu setup that uses both docker images. it's in the `old_random_backup_scripts` folder.

any fine-tunes hosted on huggingface can be used in both backends. 
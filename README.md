# gary-backend-combined

this is the backend for [gary4live](https://github.com/betweentwomidnights/gary4live), the max for live device for iterating on your existing ableton projects.

it's also the backend for gary4beatbox, a free ios/android app that does almost all the same stuff using your mic as input audio (or stable audio open small)

https://thecollabagepatch.com/redirect.html

we run this backend for free at `g4l.thecollabagepatch.com` and collect zero user data cuz that would be too much work. but if you want to run it locally, here's how.

## july 2025 update - modular architecture

this repo now uses git submodules for a cleaner, more maintainable structure:

- **stable-audio-api**: jerry (stable-audio-open-small) for instant 12-second generations
- **melodyflow**: terry (meta's melodyflow) for audio transformation 
- **gpu-queue-service**: go-based queue management for concurrent usage
- **main backend**: g4lwebsockets and core gary functionality

each submodule has its own readme if you want to dig deeper. stable-audio-api can run by itself. melodyflow can easily be modified to as well.

## gpu requirements

**minimum**: 6gb gpu ram (like rtx 3060)  
**recommended**: 12gb+ gpu ram for comfortable usage

### memory breakdown:
- **stable-audio-open-small**: ~2gb (stays loaded for speed)
- **melodyflow**: ~9gb while running (t4 usage)
- **musicgen-small**: ~2gb (cleared after generation)
- **musicgen-medium**: ~5gb (cleared after generation) 
- **musicgen-large**: ~9gb (cleared after generation)

## quick start

### prerequisites

**docker & docker-compose**:
- install docker desktop: https://www.docker.com/products/docker-desktop/
- `pip install docker-compose` (or `docker compose` depending on your install)

**huggingface token** (required for stable-audio):
- create account at https://huggingface.co
- get access to https://huggingface.co/stabilityai/stable-audio-open-small
- generate token at https://huggingface.co/settings/tokens

### installation

```bash
# 1. Clone with all submodules
git clone --recurse-submodules https://github.com/betweentwomidnights/gary-backend-combined
cd gary-backend-combined

# 2. Build infrastructure containers first (from main directory)
docker build -t thecollabagepatch/redis:optimized -f Dockerfile.redis-optimized .
docker build -t thecollabagepatch/mongo:latest -f Dockerfile.mongo .
docker build -t thecollabagepatch/g4lwebsockets:latest -f Dockerfile.g4lwebsockets .

# 3. Build gpu-queue-service
cd gpu-queue-service
docker build -t thecollabagepatch/go:dual -f Dockerfile .
cd ..

# 4. Build melodyflow  
cd melodyflow
docker build -t thecollabagepatch/melodyflow:latest -f Dockerfile.melodyflow .
cd ..

# 5. Set HF_TOKEN environment variable (required for stable-audio)
export HF_TOKEN=your_hugging_face_token_here

# 6. Build stable-audio-api
cd stable-audio-api
docker build -t thecollabagepatch/stable-gary:latest -f Dockerfile .
cd ..

# 7. Start everything
docker compose up -d

# 8. Watch the logs
docker compose logs -f
```

you should see the containers run after a few seconds of irrelevant warnings.

done! if you have trouble installing docker-compose, just ask claude (lol). i do all this in wsl for ease of use. 

## what each service does

### gary (g4lwebsockets) - port 8000
the main websockets server that handles:
- musicgen continuations and generations
- communication with max for live
- session management and audio processing

### jerry (stable-audio-api) - port 8005  
instant audio generation using stable-audio-open-small:
- 12 seconds of bpm-aware audio in under a second
- great for creating input audio for gary's continuations
- can be used standalone as an api

### terry (melodyflow) - port 8002
audio transformation using meta's melodyflow:
- transforms input audio while maintaining bpm and structure
- same-length output as input
- great for morphing gary's noisy outputs into something more interesting. accordion dnb = based

### queue service - port 8085
go-based concurrent request handling:
- prevents gpu overload from multiple users
- probably overkill for solo use but doesn't break anything
- enables sharing your backend with friends

## apple silicon (arm64) status

**currently not working** - i don't have an apple silicon machine to test with. if you're on mac and want to help debug this, hit me up in the discord: https://discord.gg/VECkyXEnAd

the arm64 dockerfiles exist but haven't been debugged fully. i had a chance to try for a few hours a while back and failed. contributions welcome!

## troubleshooting

**out of memory errors**:
- you probably need more gpu ram

**containers won't start**:
- make sure you have enough disk space (models are large)
- check `docker compose logs` for specific errors
- ensure your gpu supports cuda

**stable-audio fails to load**:
- verify your HF_TOKEN is correct
- make sure you have access to the stable-audio-open-small model

## using with gary4live

once the backend is running:

1. open gary4live max for live device
2. in `commentedout.js`, switch the configuration to localhost (see gary4live readme)
3. press the "3" button in the m4l device to connect
4. you should see "backend connected" in the electron app

## other applications

this backend also supports:
- **gary4beatbox**: ios/android app (same backend as gary4live)
- **gary-on-the-fly**: browser extension for youtube continuations
- **gary4web**: web interface (currently offline)

## development

if you want to modify or extend the backend:
- each submodule can be developed independently
- main websockets server is in `g4lwebsockets.py`
- docker-compose orchestrates everything
- check individual submodule readmes for specific dev instructions

## support

if you rly wanna learn how to use this thing, head to discord https://discord.gg/VECkyXEnAd and yell at me @kev, or go to https://youtube.com/@thepatch_dev

## related repositories

- **frontend**: https://github.com/betweentwomidnights/gary4live
- **mac frontend**: https://github.com/betweentwomidnights/gary-mac
- **training tools**: https://github.com/betweentwomidnights/train-gary
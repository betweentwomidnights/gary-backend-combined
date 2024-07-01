# contributing to gary

this file is a random explosion of thoughts right now:


# thoughts

we got 3 front-ends now. only the one for the ableton device works gr8. front-end chads plz come help with state management.

gary-on-the-fly is for the chrome extension (concurrent_gary)
gary-4-live is for ableton (g4lwebsockets)
gary-4-web is at https://thecollabagepatch.com (concurrent_gary)


i would love to do a CLI for this. there are several versions of the docker-compose.yml so that one could spin up solo instances of g4lwebsockets or concurrent_gary

you might notice CommieGary.py. that is a bad attempt at beginning to make a module for commune. if you wanna help with that, get at me.

this repo itself could be cleaned up and managed by someone after a short one-on-one with me.

there is a 'custom musicgen class' in here. it's not special at all and just uses the 'set_custom_progress_callback' that meta already provided.

other things...

jasco gonna prolly get added to this. 

also i'd love to add the open source stable audio model. we have front-ends that can be adapted to do some neat stuff im sure.




# TODO I GUESS

CLI for spinning up the various backends with ease.

finish the backend for 'train-gary' and combine it with this. https://github.com/betweentwomidnights/train-gary

massive repo organization stuff

fix concurrent_gary. it is so damn slow. prolly needs to abandon the rq worker.

// this is just here for the member berries rn
gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 2 -t 240 --graceful-timeout 30 g4lwebsockets:app

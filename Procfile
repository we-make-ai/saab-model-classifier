heroku buildpacks:clear
heroku buildpacks:add --index heroku/python
heroku ps:scale web=0
heroku ps:scale web=1
web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker --log-level warning server:app

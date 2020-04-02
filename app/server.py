import aiohttp
import asyncio
import uvicorn

from fastai2.basics import *
from fastai2.vision.all import *


from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route
import logging

from utils import *

classes = ['Saab_9000', 'Saab_900', 'Saab_9-3', 'Saab_9-5']
path = Path(__file__).parent

templates = Jinja2Templates(directory=str('app/templates'))

app = Starlette(debug=True)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
app.mount('/templates', StaticFiles(directory='app/templates'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
path = Path('./')


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
  img_data = await request.form()
  img_bytes = await (img_data['file'].read())
  pred = learn.predict(BytesIO(img_bytes))[0]
  return JSONResponse({
      'results': str(pred)
  })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        #print(__main__)
        logger.debug("starting...")
        logger.info("starting uvicorn server at port 5000")
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")


@app.route("/analyze", methods=["POST"])
async def analyze(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    context = {
        "request": request, 
        "data": predict_image_from_bytes(bytes)
    }
    return templates.TemplateResponse('show_predictions.html', context=context)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    x,y,losses = learn.predict(img)   
    return JSONResponse({
        "predictions": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[0]
        ),
        "results": [(label, prob) for label, prob in zip(learn.data.classes, map(round, (map(float, losses*100))))]   
    })

@app.route('/')
async def homepage(request):
    html_file = path / 'templates' / 'index.html'
    return templates.TemplateResponse("index.html", {"request": request})



if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

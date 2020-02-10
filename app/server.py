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
import logging


# Liegt auf gdrive von elexis.austria@gmail.com
export_file_url = 'https://we-make-ai-blog.s3.eu-de.cloud-object-storage.appdomain.cloud/saab-model-classifier.pkl'
export_file_name = 'saab-model-classifier.pkl'

classes = ['Saab_9000', 'Saab_900', 'Saab_9-3', 'Saab_9-5']
path = Path(__file__).parent

app = Starlette(debug=True)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
#app.mount('/static', StaticFiles(directory='static'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

async def download_file(url, dest):
    logger.debug('downloading_file')
    if dest.exists(): 
        logger.debug('ML Model exists, skipping download')
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = torch.load(path/export_file_name)
        defaults.device = torch.device('cpu')
        learn = load_learner(path/export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            logger.debug(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


# import pickle

# class CustomUnpickler(pickle.Unpickler):

#     def find_class(self, module, name):
#         if name == 'Manager':
#             from settings import Manager
#             return Manager
#         return super().find_class(module, name)

#pickle_data = CustomUnpickler(open('file_path.pkl', 'rb')).load()

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
path = Path('./')
#learn = load_learner(Path('saab-model-classifier.pkl'))
#learner = CustomUnpickler(open('saab-model-classifier.pkl'))
lern = setup_learner()

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

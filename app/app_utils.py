import aiohttp
import asyncio

from fastai2.basics import *
from fastai2.vision.all import *

import logging

export_file_url = 'https://drive.google.com/open?id=1yK9znhiZ4fEUQRLrUpnI_rHfq2DQ-07Q'

export_file_name = 'saab-classifier.pkl'

classes = ['Saab_9000', 'Saab_900', 'Saab_9-3', 'Saab_9-5']
path = Path(__file__).parent
path = Path('./')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(str(path))

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
        #learn = torch.load(path/export_file_name)
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

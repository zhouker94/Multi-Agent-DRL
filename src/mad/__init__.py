import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

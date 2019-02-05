import os
import dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def load_env(filename='.env'):
    dotenv_path = os.path.join(BASE_DIR, '..', filename)
    dotenv.load_dotenv(dotenv_path)

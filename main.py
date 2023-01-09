## Create Virtaul Env       ## python -m venv /path/to/mnist_web_venv
## Activate Virtual Env     ## path/to/mnist_web_venv/Scripts/activate    ## C:/users/ayush/mnist_web_venv/Scripts/activate
## Upgrade pip              ## pip install --upgrade pip
## Install dependencies     ## pip install -r requirements.txt
## If Python isn't on your Windows path, you may need to type out the full path to pyinstaller to get it to run.

from flask_app.routers import app

if __name__ == '__main__':
    app = app
    app.run()

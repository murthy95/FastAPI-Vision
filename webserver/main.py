import base64
import io
import json
import os
import time
import uuid

import numpy as np
from PIL import Image
import redis

from fastapi import FastAPI, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.requests import Request

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))
CLIENT_SLEEP = float(os.environ.get("CLIENT_SLEEP"))

IMAGE_QUEUE = os.environ.get("IMAGE_QUEUE")

IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: Request, img_file: bytes = File(...)):
    data = {"success": False}

    if request.method == "POST":
        image = Image.open(io.BytesIO(img_file))
        width, height = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
        image = np.asarray(image).astype(np.float32)
        image = image.copy(order="C")

        # Generate an ID for the classification then add the classification ID + image to the queue
        k = str(uuid.uuid4())

        image = base64.b64encode(image).decode("utf-8")
        d = {"id": k, "image": image, "width": width, "height": height}
        db.rpush(IMAGE_QUEUE, json.dumps(d))

        # Keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1

            # Attempt to grab the output predictions
            output = db.get(k)

            # Check to see if our model has classified the input image
            if output is not None:
                # Add the output predictions to our data dictionary so we can return it to the client
                # output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                # Delete the result from the database and break from the polling loop
                db.delete(k)
                break

            # Sleep for a small amount to give the model a chance to classify the input image
            time.sleep(CLIENT_SLEEP)

            # Indicate that the request was a success
            data["success"] = True
            data["tries"] = num_tries
        else:
            raise HTTPException(
                status_code=400,
                detail="Request failed after {} tries".format(CLIENT_MAX_TRIES),
            )

    # Return the data dictionary as a JSON response
    return data

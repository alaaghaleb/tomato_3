from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load your TensorFlow model
model = tf.keras.models.load_model("inception.model.h5")


@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    try:
        # Ensure the file is an image
        allowed_formats = ('.png', '.jpg', '.jpeg')
        if not image_file.filename.lower().endswith(allowed_formats):
            raise HTTPException(status_code=400,
                                detail='Unsupported file format. Only PNG and JPEG images are supported.')

        # Read the image file
        contents = await image_file.read()

        # Convert image data to PIL Image
        pil_image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        pil_image = pil_image.resize((224, 224))  # Resize image if needed
        x = np.array(pil_image)
        x = x / 255.0  # Normalize pixel values
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]

        # Map prediction to disease names
        disease_names = {
            0: "Bacterial Spot",
            1: "Early Blight",
            2: "Late Blight",
            3: "Leaf Mold",
            4: "Septoria Leaf Spot",
            5: "Spider Mites",
            6: "Target Spot",
            7: "Yellow Leaf Curl Virus",
            8: "Mosaic Virus",
            9: "Healthy"
        }
        predicted_disease = disease_names.get(pred_class, "Unknown")

        if pred_class not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return JSONResponse(content={'wrongprediction': "Please enter a good image"}, status_code=200)
        else:
            return JSONResponse(content={'prediction': predicted_disease}, status_code=200)

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000)

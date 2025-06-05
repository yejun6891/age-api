# Age Prediction API

This project provides a Flask based API for estimating the age of a face in an image using the **demoage** model. It is ready to be deployed on [Render](https://render.com/).

## Setup

```bash
pip install -r requirements.txt
```

## Running locally

```bash
python app.py
```

The server will start on port 5000.

### Render

Render will use the included `Procfile`. Simply create a new web service and point it to this repository.

## Usage

`POST /predict`

Send an image either as a multipart/form-data file parameter named `image` or as JSON containing a base64 encoded string in the `image` field.

**Example using curl with file:**
```bash
curl -X POST -F image=@face.jpg http://localhost:5000/predict
```

**Example using curl with base64 JSON:**
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image": "<base64 string>"}' http://localhost:5000/predict
```

The response will be:

```json
{
  "age": 25,
  "explanation": "AI는 이마 주름과 눈가 음영을 근거로 판단했습니다."
}
```

`age` is the predicted age and `explanation` summarizes the Grad-CAM result.

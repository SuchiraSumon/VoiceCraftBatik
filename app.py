from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from scipy.spatial.distance import euclidean
import os
import numpy as np
import librosa
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import io
import base64
import random

app = FastAPI()

# Serve static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Directories for batik types
batik_directory = r'C:/Users/a3154/OneDrive/Desktop/FYP1/Dataset 2/Batik'
batikblok_directory = r'C:/Users/a3154/OneDrive/Desktop/FYP1/Dataset 2/BatikBlok'
batiklukis_directory = r'C:/Users/a3154/OneDrive/Desktop/FYP1/Dataset 2/BatikLukis'
batikskrin_directory = r'C:/Users/a3154/OneDrive/Desktop/FYP1/Dataset 2/BatikSkrin'

# Previous selections to avoid re-selection
previous_selections = []
previous_selections_blok = []
previous_selections_lukis = []
previous_selections_skrin = []

# Load batik images and their extracted features
def load_batik_images(directory):
    batik_features = []
    batik_labels = []
    batik_images = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(directory, file_name)
            image = load_img(file_path)
            image_array = img_to_array(image)

            edge_count = compute_edge_detection(image_array)
            color_histogram = compute_color_histogram(image_array)
            fourier_value = compute_fourier_transform(image_array)

            batik_features.append([edge_count, color_histogram, fourier_value])
            batik_labels.append(file_name)
            batik_images[file_name] = image_array

    return batik_features, batik_labels, batik_images

# Edge detection function
def compute_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_uint8 = np.uint8(gray * 255)
    edges = cv2.Canny(gray_uint8, 100, 200)
    edge_count = np.sum(edges > 0)
    return edge_count

# Color histogram function
def compute_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return np.sum(hist)

# Fourier transform function
def compute_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    magnitude_spectrum = np.abs(f_transform)
    fourier_value = np.sum(magnitude_spectrum)
    return fourier_value

# Extract audio features function
def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.nan
    if pitches.size > 0 and magnitudes.size > 0:
        valid_pitches = pitches[magnitudes > np.median(magnitudes)]
        if valid_pitches.size > 0:
            pitch = np.max(valid_pitches)
        pitch = pitch.item() if isinstance(pitch, np.ndarray) else float(pitch)

    rms_values = librosa.feature.rms(y=y)
    rms = np.sqrt(np.mean(rms_values**2))
    if isinstance(rms, np.ndarray):
        rms = rms[0]
    rms = float(rms) if np.isfinite(rms) else np.nan

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) and tempo.size > 0 else float(tempo) if tempo > 0 else np.nan

    return float(pitch), float(rms), float(tempo)

# Function to find most accurate batik
def find_two_most_accurate_batik(audio_features, batik_features, batik_labels):
    distances = []
    for i, batik_feature in enumerate(batik_features):
        if batik_labels[i] in previous_selections:
            continue

        distance = euclidean(audio_features, batik_feature)
        random_noise = random.uniform(0, 0.05)
        distance += random_noise
        distances.append((distance, batik_labels[i]))

    distances.sort(key=lambda x: x[0])
    best_match_1 = distances[0][1] if len(distances) > 0 else None
    best_match_2 = distances[1][1] if len(distances) > 1 else None

    if best_match_1:
        previous_selections.append(best_match_1)
    if best_match_2:
        previous_selections.append(best_match_2)

    return best_match_1, best_match_2

# Convert images to base64 for rendering in HTML
def convert_image_to_base64(image):
    plt.imshow(image.astype('uint8'))
    plt.axis('off')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    return base64.b64encode(img_data.getvalue()).decode()

# Load batik images and features at the start of the app
batik_features, batik_labels, batik_images = load_batik_images(batik_directory)
batikblok_features, batikblok_labels, batikblok_images = load_batik_images(batikblok_directory)
batiklukis_features, batiklukis_labels, batiklukis_images = load_batik_images(batiklukis_directory)
batikskrin_features, batikskrin_labels, batikskrin_images = load_batik_images(batikskrin_directory)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/choosegbatik.html", response_class=HTMLResponse)
async def choosegbatik(request: Request):
    return templates.TemplateResponse("choosegbatik.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/gbatik.html", response_class=HTMLResponse)
async def gbatik_form(request: Request):
    return templates.TemplateResponse("gbatik.html", {"request": request})

@app.post("/gbatik.html", response_class=HTMLResponse)
async def gbatik(request: Request):

    audio_path = "./uploaded_files/audio_recording.wav"

    # Ensure the file exists before proceeding 
    if not os.path.exists(audio_path): 
        return HTMLResponse(content="File not found", status_code=404)

    pitch, volume, tempo = extract_audio_features(audio_path)
    audio_features = [pitch, volume, tempo]
    best_match_1, best_match_2 = find_two_most_accurate_batik(audio_features, batik_features, batik_labels)

    images = []
    for match in [best_match_1, best_match_2]:
        if match:
            image_data = convert_image_to_base64(batik_images[match])
            images.append((match, image_data))

    os.remove(audio_path)
    return templates.TemplateResponse("gbatik.html", {"request": request, "audio_features": audio_features, "images": images})

@app.get("/batikalternatif.html", response_class=HTMLResponse)
async def batikalternatif_form(request: Request):
    return templates.TemplateResponse("batikalternatif.html", {"request": request})

@app.get("/batiklukis.html", response_class=HTMLResponse)
async def batiklukis_form(request: Request):
    return templates.TemplateResponse("batiklukis.html", {"request": request})

@app.get("/batikskrin.html", response_class=HTMLResponse)
async def batikskrin_form(request: Request):
    return templates.TemplateResponse("batikskrin.html", {"request": request})

@app.get("/batikblok.html", response_class=HTMLResponse)
async def batikblok_form(request: Request):
    return templates.TemplateResponse("batikblok.html", {"request": request})

@app.get("/gbatikblok.html", response_class=HTMLResponse)
async def gbatikblok_form(request: Request):
    return templates.TemplateResponse("gbatikblok.html", {"request": request})

@app.post("/gbatikblok.html", response_class=HTMLResponse)
async def gbatikblok(request: Request):

    audio_path = "./uploaded_files/audio_recording.wav"

    # Ensure the file exists before proceeding 
    if not os.path.exists(audio_path): 
        return HTMLResponse(content="File not found", status_code=404)

    pitch, volume, tempo = extract_audio_features(audio_path)
    audio_features = [pitch, volume, tempo]
    best_match_1, best_match_2 = find_two_most_accurate_batik(audio_features, batikblok_features, batikblok_labels)

    images = []
    for match in [best_match_1, best_match_2]:
        if match:
            image_data = convert_image_to_base64(batikblok_images[match])
            images.append((match, image_data))

    os.remove(audio_path)
    return templates.TemplateResponse("gbatikblok.html", {"request": request, "audio_features": audio_features, "images": images})

@app.get("/gbatiklukis.html", response_class=HTMLResponse)
async def gbatiklukis_form(request: Request):
    return templates.TemplateResponse("gbatiklukis.html", {"request": request})

@app.post("/gbatiklukis.html", response_class=HTMLResponse)
async def gbatiklukis(request: Request):
    
    audio_path = "./uploaded_files/audio_recording.wav"

    # Ensure the file exists before proceeding 
    if not os.path.exists(audio_path): 
        return HTMLResponse(content="File not found", status_code=404)

    pitch, volume, tempo = extract_audio_features(audio_path)
    audio_features = [pitch, volume, tempo]
    best_match_1, best_match_2 = find_two_most_accurate_batik(audio_features, batiklukis_features, batiklukis_labels)

    images = []
    for match in [best_match_1, best_match_2]:
        if match:
            image_data = convert_image_to_base64(batiklukis_images[match])
            images.append((match, image_data))

    os.remove(audio_path)
    return templates.TemplateResponse("gbatiklukis.html", {"request": request, "audio_features": audio_features, "images": images})

@app.get("/gbatikskrin.html", response_class=HTMLResponse)
async def gbatikskrin_form(request: Request):
    return templates.TemplateResponse("gbatikskrin.html", {"request": request})

@app.post("/gbatikskrin.html", response_class=HTMLResponse)
async def gbatikskrin(request: Request):
    
    audio_path = "./uploaded_files/audio_recording.wav"

    # Ensure the file exists before proceeding 
    if not os.path.exists(audio_path): 
        return HTMLResponse(content="File not found", status_code=404)


    pitch, volume, tempo = extract_audio_features(audio_path)
    audio_features = [pitch, volume, tempo]
    best_match_1, best_match_2 = find_two_most_accurate_batik(audio_features, batikskrin_features, batikskrin_labels)

    print(f"Data extracted: {pitch} + {volume} + {tempo}")

    images = []
    for match in [best_match_1, best_match_2]:
        if match:
            image_data = convert_image_to_base64(batikskrin_images[match])
            images.append((match, image_data))

    os.remove(audio_path)
    return templates.TemplateResponse("gbatikskrin.html", {"request": request, "audio_features": audio_features, "images": images})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)): 
    try: 
        contents = await file.read() 
        with open(f"uploaded_files/{file.filename}", "wb") as f:
            f.write(contents)
            return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})
    except Exception as e: 
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
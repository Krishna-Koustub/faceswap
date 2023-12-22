from fastapi import FastAPI, File, UploadFile,Query
from fastapi.responses import FileResponse
import os
import asyncio
import aiohttp
from datetime import datetime
app = FastAPI()

async def download_file(url, destination):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
            with open(destination, 'wb') as f:
                f.write(content)

async def run_face_swapper(target_video_path, source_image_path,output_video_path):
    if not os.path.exists("roop"):
        os.system("git clone https://github.com/s0md3v/roop.git")

    os.chdir("roop")

    if not os.path.exists("requirements.txt"):
        os.system("pip install -r requirements.txt")

    model_path = "models/inswapper_128.onnx"
    if not os.path.exists(model_path):
        os.system("mkdir models")
        await download_file("https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx", model_path)

    if "onnxruntime" in os.popen("pip show onnxruntime").read() or "onnxruntime-gpu" in os.popen("pip show onnxruntime-gpu").read():
        os.system("pip uninstall onnxruntime onnxruntime-gpu -y")

    if "torch" not in os.popen("pip show torch").read():
        os.system("pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118")

    if "onnxruntime-gpu" not in os.popen("pip show onnxruntime-gpu").read():
        os.system("pip install onnxruntime-gpu")

    os.system(f"python run.py --target {target_video_path} --source {source_image_path} -o {output_video_path} --frame-processor face_swapper face_enhancer")

@app.post("/face-swapper")
async def face_swapper_route(source_image: UploadFile = File(...),q:str=Query(...,title="Gender")):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    if(q=="Male"):
        target_video_path = f"uploads/static/male.mp4"

    else:
        target_video_path = f"uploads/static/female.mp4"
    newname=datetime.now().strftime("%Y%m%d%H%M%S")

 
    filename = f"{newname}{source_image.filename}"
 



    source_image_path = f"uploads/user_uploads/{filename}"
    filename_without_extension = os.path.splitext(filename)[0]
    output_video_path = f"uploads/output/{filename_without_extension}.mp4"
    # output_video_path = f"uploads/output/.mp4"

    # Save the uploaded files
    # with open(target_video_path, "wb") as target_file:
    #     target_file.write(target_video.file.read())

    with open(source_image_path, "wb") as source_file:
        source_file.write(source_image.file.read())

    # Run the face swapper function asynchronously
    await run_face_swapper(target_video_path, source_image_path, output_video_path)

    # Return the swapped video as a response
    return FileResponse(output_video_path, media_type="video/mp4", filename="swapped.mp4")
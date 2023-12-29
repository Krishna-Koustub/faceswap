from fastapi import FastAPI, File, UploadFile,Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import asyncio
import aiohttp
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

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


    if "torch" not in os.popen("pip show torch").read():
        os.system("pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118")

    if "onnxruntime-gpu" not in os.popen("pip show onnxruntime-gpu").read():
        os.system("pip install onnxruntime-gpu")
    print('======================+++++')
    print(source_image_path)
    print('======================+++++')
    print('======================+++++')
    print(target_video_path)
    print('======================+++++')
    print('======================+++++')
    print(output_video_path)
    print('======================+++++')

    command=f"python3 run.py --target /home/krishna/Documents/faceswap/app/uploads/static/male.mp4 --source /home/krishna/Documents/faceswap/app/uploads/user_uploads/20231229144627ujwal.jpeg -o /home/krishna/Documents/faceswap/app/uploads/output/vid.mp4 --frame-processor face_swapper face_enhancer"
    logging.debug(f"Executing command: {command}")
    os.system(command)

app.mount("/uploads", StaticFiles(directory="./app/uploads"), name="uploads")



@app.post("/face-swapper")
async def face_swapper_route(source_image: UploadFile = File(...),q:str=Query(...,title="Gender")):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    print(os.curdir)
    if(q=="Male" or q=="male"):
        target_video_path = '/uploads/static/male.mp4'

    else:
        target_video_path = '/uploads/static/female.mp4'
    newname=datetime.now().strftime("%Y%m%d%H%M%S")

 
    filename = f"{newname}{source_image.filename}"
 



    source_image_path = f"/app/uploads/user_uploads/{filename}"
    filename_without_extension = os.path.splitext(filename)[0]
    output_video_path = f"/uploads/output/{filename_without_extension}.mp4"
    # output_video_path = f"uploads/output/.mp4"
    
    # Save the uploaded files
    # with open(target_video_path, "wb") as target_file:
    #     target_file.write(target_video.file.read())
    
    with open(source_image_path, "wb") as source_file:
        source_file.write(source_image.file.read())

    # Run the face swapper function asynchronously
    await run_face_swapper(target_video_path, source_image_path, output_video_path)

    # Return the swapped video as a response
    return True#FileResponse(output_video_path, media_type="video/mp4")



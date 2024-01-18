from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler, AutoencoderKL

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
# Configure the pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    #variant="fp16"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe.to(device)


app = Flask(__name__)

# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read())
#         return encoded_image.decode("utf-8")

def base64_to_image(base64_string, output_path):
    # Decode Base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create an in-memory binary stream
    image_stream = io.BytesIO(image_data)

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_stream)

    # Save the image to the specified output path
    image.save(output_path)

def manipulate_image(image_pil, style, strength,lora_scale_slider):
    """PILLOW IMAGE, and PROMPT"""
    if style == "anime detailer":
        pipe.load_lora_weights('Linaqruf/anime-detailer-xl-lora', weight_name='anime-detailer-xl.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "shonen anime character, matching facial features and expressions, exact matching outfit, beard if any,  best quality, similar hairstyle with exact hair colour"
        negative_prompt =  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "gangsta":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.gangsta', weight_name='gangsta.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "gangsta anime character, moll if any, clarity, realistic, best quality, beard if any, chain on neck, bracelate in hand, ear lobe piercing, phone in hand if any, same hairstyle and color"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, bad eyes, distorded body parts, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "anime":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.anime', weight_name='anime.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt =  "Anime, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, bad hands, text, error, missing fingers, bad eyes, distorded body parts, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "3d animated movie still":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.3d-animated-movie-still', weight_name='3d animated movie still.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "3d-anime movie character, realistic, high resolution, high quality, clarity, same facial expression and features, accessories like phone in hand if any3d-anime movie character, realistic, high resolution, high quality, clarity, same facial expression and features, accessories like phone in hand if any"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, bad eyes, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "vampire":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.vampire', weight_name='vampire.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "vampire, gender based on image, realistic, high resolution, high quality, clarity, gothic fashion clothing"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=6,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "pixar style":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.pixar-style', weight_name='pixar-style.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "pixar style, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "dreamscape":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.dreamscape', weight_name='dreamscape.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "dreamscape based, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "mohawk":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.mohawk', weight_name='mohawk.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt =  "mohawk based, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "pirate":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.pirate', weight_name='pirate.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "anime pirate, gender based i.e. male female based on image, realistic, high resolution, high quality, clarity, pirate hat, pirate ship"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, distorted fingers, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=6,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "cartoon":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.cartoon', weight_name='cartoon.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "cartoon character, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "fit":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.fit', weight_name='fit.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "Anime character with fit body, gender based i.e. female masculine and male masculine based on image, realistic, high resolution, high quality, clarity"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "demon":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.demon', weight_name="demon.safetensors")
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = " Anime, Demon character, gender based on image, realistic, high resolution, high quality, clarity, fire spirits coming out of hand"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "pixel art":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.pixel-art', weight_name='pixel art.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = " pixel art, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "ultra realistic illustration":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.ultra-realistic-illustration', weight_name='ultra realistic illustration.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "realistic, gender based on image, realistic, high resolution, high quality, clarity, mobile in hand if any"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "comic portrait":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.comic-portrait', weight_name='comic portrait.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "comic anime character, black and white, gender based on image, realistic, high resolution, high quality, clarity, mobile in hand if any"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "rich":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.rich', weight_name='rich.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "rich, accessories, gender based on image, realistic, high resolution, high quality, clarity, sports cars"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "90s anime":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.90s-anime', weight_name='90s anime.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "90s anime, gender based on image, high resolution, high quality, clarity"
        negative_prompt = "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "makeup":
        pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.makeup', weight_name='makeup.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt =  "makeup, accessories, gender based on image, realistic, high resolution, high quality, clarity"
        negative_prompt =  "lowres, bad anatomy, distorted body, text, error, missing fingers, blur clothes, distorted fingers, missing hand, bad body parts, distorted hand, bad face, blur background, distorted background, bad eyes,distored image, distorded body parts, extra fingers, distorted hand, bad hairs, inappropriate eyebrows, bad lips, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blur, distorted"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)
    
    elif style == "concept art":
        pipe.load_lora_weights('KappaNeuro/syd-mead-style', weight_name='Syd Mead Style.safetensors')
        pipe.fuse_lora(lora_scale=lora_scale_slider)
        prompt = "CONCEPT ART BY SYD MEAD, RETRO FUTURISTIC, FLAT COLORS, best quality"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        image = pipe(prompt, negative_prompt=negative_prompt, image = image_pil, strength = strength, guidance_scale=12,num_inference_steps=50).images[0]
        pipe.unfuse_lora()
        return np.array(image)

def numpy_to_base64(image_np):
    # Convert the NumPy array to PIL Image
    manipulated_image = Image.fromarray(image_np.astype('uint8'))

    # Create an in-memory binary stream for the manipulated image
    manipulated_image_stream = io.BytesIO()
    manipulated_image.save(manipulated_image_stream, format='PNG')

    # Encode the manipulated image to Base64
    manipulated_base64 = base64.b64encode(manipulated_image_stream.getvalue()).decode('utf-8')
    return manipulated_base64

@app.route('/img2img', methods=['POST'])
def pipeline():
    # Get data from the request
    data = request.get_json()

    # Extract image and prompt from the data
    base64_image = data.get('image')
    # prompt = data.get('prompt')
    style = data.get("style")
    strength = float(data.get("strength"))
    lora_scale_slider = int(data.get("lora_scale_slider"))
    # Decode Base64 image to NumPy array
    image_data = base64.b64decode(base64_image)
    image_np = np.array(Image.open(io.BytesIO(image_data)))
    image_pil = Image.fromarray(image_np)

    # Perform dummy manipulation on the image
    manipulated_image_np = manipulate_image(image_pil, style, strength,lora_scale_slider)

    # Convert the manipulated image back to Base64
    manipulated_base64 = numpy_to_base64(manipulated_image_np)

    # Return the manipulated image as Base64 along with the prompt
    response_data = {
        'manipulated_image': manipulated_base64
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)


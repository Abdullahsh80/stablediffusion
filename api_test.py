import requests
import base64
import io
from PIL import Image

# Function to convert image to Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")
    
def base64_to_image(base64_string):
    # Decode Base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create an in-memory binary stream
    image_stream = io.BytesIO(image_data)

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_stream)

    # Save the image to the specified output path
    # image.save(output_path)
    return image


# Replace 'your_image_path.jpg' with the actual path to your image file
image_path = r'C:\Users\abdul\OneDrive\Desktop\Projects\sdpipeline\Dragon-Ball-Z.jpeg'

# Convert the image to Base64
base64_image = image_to_base64(image_path)

# Define the API endpoint
api_endpoint = 'http://127.0.0.1:5000/img2img'

# Prepare the payload with image and prompt
payload = {
    'image': base64_image,
    "style": "anime",
    "strength": 0.8,
    "lora_scale_slider": 2

}

# Send the POST request to the API
response = requests.post(api_endpoint, json=payload)

img = base64_to_image(response.json()['manipulated_image'])

img.show()
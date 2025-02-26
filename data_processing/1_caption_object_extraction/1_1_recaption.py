import os
import json
from pathlib import Path
from openai import OpenAI

# Function to initialize OpenAI client with API key and base URL
def initialize_openai_client(api_key, api_base):
    return OpenAI(api_key=api_key, base_url=api_base)

# Function to generate a caption for an image
def generate_caption(client, image_path):
    try:
        with open(image_path, "rb") as img_file:
            # Assuming this calls a custom API to upload the image and return a caption
            response = client.Image.create(file=img_file, purpose="answers")

        # Assuming the caption is stored in the 'caption' field of the response
        caption = response['data'][0]['caption']
        return caption

    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None

# Function to process all image files in the source directory and save captions as JSON files
def process_images(client, source_img_dir, output_dir):
    # Get all image files from the source directory (assuming .jpg format)
    image_files = [f for f in Path(source_img_dir).glob("*.jpg")]

    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image and save the generated caption as a JSON file
    for image_path in image_files:
        # Get the relative path of the image
        img_path_relative = f"./{os.path.relpath(image_path, '../data_processing/')}"

        # Generate the caption for the image
        caption = generate_caption(client, image_path)
        
        if caption:
            # Construct the JSON data to be saved
            result = {
                "img_path": img_path_relative,
                "caption": caption
            }

            # Save the JSON data to a file
            json_filename = image_path.stem + ".json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w') as json_file:
                json.dump(result, json_file, ensure_ascii=False, indent=4)
            
            print(f"Caption saved for {image_path}: {json_filename}")
        else:
            print(f"Skipping {image_path}, caption not generated.")

# Main function
def main():
    # Set your OpenAI API key and base URL
    openai_api_key = ""  # Replace with your OpenAI API key
    openai_api_base = ""  # Replace with your custom API base URL

    # Input image folder and output folder paths
    source_img_dir = "../data_processing/assets/0_source_img"
    output_dir = "../data_processing/assets/1_caption_object_extraction/1_1_recaption"

    # Initialize OpenAI client
    client = initialize_openai_client(openai_api_key, openai_api_base)

    # Process all images and generate corresponding JSON files
    process_images(client, source_img_dir, output_dir)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

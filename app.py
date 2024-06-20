import os
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPTokenizer
import torch
import ModelLoader
import Pipeline

app = Flask(__name__)
DEVICE = "cpu"
ALLOW_MPS = False

if (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Initialize tokenizer and models
tokenizer = CLIPTokenizer(
    "/Users/urvisinghal/Desktop/Text to image/data/vocab.json",
    merges_file="/Users/urvisinghal/Desktop/Text to image/data/merges.txt"
)
model_file = "/Users/urvisinghal/Desktop/Text to image/data/v1-5-pruned-emaonly.ckpt"
models = ModelLoader.preload_models_from_standard_weights(model_file, DEVICE)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Text to Image</title>
    <h1>Text to Image</h1>
    <form id="text-form">
        <input type="text" id="text-input" placeholder="Enter text prompt">
        <input type="submit" value="Generate">
    </form>
    <img id="generated-image" style="display:none;">
    <script>
        document.getElementById('text-form').addEventListener('submit', function(event) {
            event.preventDefault();
            const text = document.getElementById('text-input').value;

            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                console.log('Response:', data);  // Debug: Log the response data
                const imageUrl = data.image_url;
                const imageElement = document.getElementById('generated-image');
                imageElement.src = imageUrl;
                imageElement.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    </script>
    '''

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('text', '')

        print(f"Received prompt: {prompt}")  # Debug: Print received prompt

        uncond_prompt = ""  
        do_cfg = True
        cfg_scale = 8  # min: 1, max: 14
        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        # Generate image
        output_image = Pipeline.generate(
            uncond_prompt=uncond_prompt,
            input_image=None,
            strength=0.8,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
            prompt=prompt,
        )

        image = Image.fromarray(output_image)
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        image_path = os.path.join(static_dir, 'generated_image.png')
        image.save(image_path)

        print(f"Image saved to {image_path}")  # Debug: Print image path

        return jsonify({'image_url': '/' + image_path})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

import json
import os
from PIL import Image
import base64
from io import BytesIO

# Step 1: Load the annotation IDs from the new_annotations.json file
annotations_file = 'new_annotations.json'
with open(annotations_file, 'r') as f:
    annotations_data = json.load(f)

# Assuming there's only one domain or you want to process 'test_domain'
annotation_ids1 = annotations_data.get('test_domain', [])
annotation_ids2 = annotations_data.get('test_site', [])
annotation_ids3 = annotations_data.get('test_task', [])
annotation_ids = annotation_ids1 + annotation_ids2 + annotation_ids3

# Step 2: For each annotation ID, load the corresponding JSON from the repository
repository_path = '../alpaca_datasets/mind2web_screenshots'

for annotation_id in annotation_ids:
    json_path = os.path.join(repository_path, f'{annotation_id}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            actions = json.load(f)
        
        # Step 3: Iterate through the actions in the JSON
        for action in actions:
            screenshot_data = action['before']['screenshot']
            if screenshot_data.startswith('/9j/'):
                # Decode the base64 image data
                image_data = base64.b64decode(screenshot_data)
                image = Image.open(BytesIO(image_data))
                
                # Step 4: Crop the image to the top 1280x1700 and save it
                cropped_image = image.crop((0, 0, 1280, 1700))
                
                # Create a directory for the annotation ID if it doesn't exist
                save_path = os.path.join('data', 'screenshots', 'mind2webnew', annotation_id)
                os.makedirs(save_path, exist_ok=True)
                
                # Save the cropped image
                action_uid = action['action_uid']
                cropped_image.save(os.path.join(save_path, f'{action_uid}.jpeg'), 'JPEG')

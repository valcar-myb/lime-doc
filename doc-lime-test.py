import os
import io
import glob
import time
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from skimage.segmentation import slic
#from lime import lime_image

from lime.lime_image import LimeImageExplainer


OUTPUT_PATH = "lime-doc-test-output"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")


def dit_inference(img):
    filename = str(time.time()) + ".png"
    image = Image.fromarray(img, mode='RGB')
    image.save(os.path.join(OUTPUT_PATH, filename))
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits, filename


def dit_classification_fn(images):
    output = []
    for img in images:
        logits, filename = dit_inference(img)
        predicted_class_idx = logits.argmax(-1).item()
        output.append(np.array(logits.tolist()[0]))
        print(filename, model.config.id2label[predicted_class_idx])
    return np.array(output)



img_path = "document-classification-test/img_test/*"

test_images = []

explainer = LimeImageExplainer()

for image in glob.glob(img_path):
  with open(image, "rb") as f:
    test_image_bytes = f.read()
    image = Image.open(io.BytesIO(test_image_bytes))
    image_array = np.array(image)
    test_images.append(image_array)

explanation = explainer.explain_instance(test_images[0],
                                         dit_classification_fn, # classification function
                                         top_labels=3,   # Solo la classe con la probabilità più alta
                                         hide_color=255, # Colore di sostituzione: bianco
                                         num_samples=100,
                                         ocr_segmentation=True,
                                         segmentation_fn=lambda x: slic(x, n_segments=25))  # Segmentazione con 50 superpixel

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pickle

# Save the object to a file
with open('explanation-'+str(time.time())+'.pkl', 'wb') as file:
    pickle.dump(explanation, file)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2000, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2000, hide_rest=True)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.show()

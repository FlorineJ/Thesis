# Visu
import random
import cv2 as cv
import matplotlib.pyplot as plt
import os

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

def random_visu_prediction(dataset_dicts, metadata, predictor, plot_dir, num_images, threshold=0.6):
    for d in random.sample(dataset_dicts, num_images):
        im = cv.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        
        # Filter out predictions below the threshold
        instances = outputs["instances"]
        high_conf_indices = instances.scores >= threshold
        filtered_instances = instances[high_conf_indices]
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(filtered_instances.to("cpu"))
        
        # Display the image
        plt.figure(figsize=(20, 20))
        plt.imshow(out.get_image()[:, :, :])
        plt.show()
        
        # Save the image with annotations
        output_filename = os.path.join(plot_dir, "visu_inference_" + d["file_name"].split("/")[-1])
        cv.imwrite(output_filename, out.get_image()[:, :, ::-1])

# Example usage:
# predictor = DefaultPredictor(cfg)
# random_visu_prediction(dataset_dicts, metadata, predictor, "output_directory", 20, threshold=0.6)

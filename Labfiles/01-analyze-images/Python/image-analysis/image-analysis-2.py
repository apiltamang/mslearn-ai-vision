# import namespaces
import os
import sys

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures, ImageAnalysisResult
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.ai.vision.imageanalysis.models import ImageAnalysisResult

def main():
    load_dotenv()

    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')

    cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))
    # Get image
    image_file = 'images/building.jpg'

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

    with open(image_file, "rb") as f:
        image_data = f.read()

    result = cv_client.analyze(image_data=image_data, visual_features=
                            [VisualFeatures.CAPTION,
                               VisualFeatures.DENSE_CAPTIONS,
                               VisualFeatures.TAGS,
                               VisualFeatures.OBJECTS,
                               VisualFeatures.PEOPLE])

    print("\nAnalysis complete. Result: ", result)

    if result.caption is not None:
        print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

    if result.dense_captions is not None:
        for caption in result.dense_captions.list:
            print(" Dense Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

    #draw_bounding_boxes(result, "images/street.jpg")
    detect_people(result, image_file)
def draw_bounding_boxes(result: ImageAnalysisResult, image_file: str) -> None:
    # draw bounding boxes around objects in the image
    # Get objects in the image
    if result.objects is not None:
        print("\nObjects in image:")

        # Prepare image for drawing
        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for detected_object in result.objects.list:
            # Print object name
            print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name,
                                                     detected_object.tags[0].confidence * 100))

            # Draw object bounding box
            r = detected_object.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)
            plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

        # Save annotated image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'objects.jpg'
        fig.savefig(outputfile)
        print('  Results saved in', outputfile)


# detect and locate people in the image
def detect_people(result: ImageAnalysisResult, image_file: str) -> None:
    # Get people in the image
    if result.people is not None:
        print("\nPeople in image:")

        # Prepare image for drawing
        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for person in result.people.list:
            # Print person name
            # print(" {} (confidence: {:.2f}%)".format(person.name, person.confidence * 100))

            # Draw person bounding box
            r = person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)
            #plt.annotate(person.name, (r.x, r.y), backgroundcolor=color)

        # Save annotated image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'people.jpg'
        fig.savefig(outputfile)
        print('  Results saved in', outputfile)

if __name__ == '__main__':
    main()
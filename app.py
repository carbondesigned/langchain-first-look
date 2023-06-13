from dotenv import load_dotenv, find_dotenv
from transformers import AutoProcessor, BlipForConditionalGeneration
from langchain import PromptTemplate, LLMChain, OpenAI
from PIL import Image
import requests

load_dotenv(find_dotenv())


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def img2text(url):
    # image_to_text = pipeline(
    #     "text-to-image", model="Salesforce/blip-image-captioning-base")
    
    image = Image.open(requests.get(url, stream=True).raw)
    
    text = "picture of: "
    
    inputs = processor(images=image, text=text, return_tensors="pt")

    outputs = model(**inputs)
    
    print(outputs)

img2text("http://images.cocodataset.org/val2017/000000039769.jpg")


# def generate_story(scenario):
#     template = """
#     You are a story teller:
#     You can generate a short story based on a simple narrative, the story should be no more than 20 words:

#     CONTEXT: {scenario}
#     STORY:
#     """

#     prompt = PromptTemplate(template=template, input_variables=["scenario"])

#     story_llm = LLMChain(
#         llm=OpenAI(
#             model="gpt-3.5-turbo", temperature=1, prompt=prompt, verbose=True
#         )
#     )

#     story = story_llm.predict(scenario=scenario)

#     print(story)
#     return story


# scenario = img_to_text("img.png")
# story = generate_story(scenario)

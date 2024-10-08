from openai import OpenAI
from PIL import Image
import os
import base64
import json
from tqdm import tqdm
import traceback


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_gif(image_path):
    frame = Image.open(image_path)
    nframes = 0
    encoded = []
    
    # Ensure the cache directory exists
    cache_dir = "feedback_fewshot/.cache/"
    os.makedirs(cache_dir, exist_ok=True)
    
    while True:
        try:
            image_file = os.path.join(cache_dir, os.path.basename(image_path) + "-" + str(nframes) + ".jpg")
            frame.convert('RGB').save(image_file)
            with open(image_file, "rb") as image:
                encoded.append(base64.b64encode(image.read()).decode('utf-8'))
            nframes += 1
            if nframes == 8: break
            frame.seek(nframes)
        except EOFError:
            break
    
    return encoded


def chat_with_openai_rf(image_paths, gif_paths, text_prompts):
    api_key = '' ##add_you_openai_api_key
    base64_image8 = encode_image("feedback_fewshot/assembly_24_test.png")
    base64_image8_out = encode_gif("feedback_fewshot/assembly_checkpoint_276_out_3.gif")
    # print(len(base64_image8_out))
    base64_image7 = encode_image("feedback_fewshot/assembly.png")
    base64_image7_out = encode_gif("feedback_fewshot/assembly_out_1iter_1.gif")
    base64_image6_out = encode_gif("feedback_fewshot/basketball_out_1iter_00.gif")
    base64_image6 = base64_image6_out[0]
    base64_image5_out = encode_gif("feedback_fewshot/hammer_checkpoint_310_out_0iter_7.gif")
    base64_image5 = base64_image5_out[0]
    # Initialize a conversation with the system role description
    print("Feedback started")

    # Initialize a conversation with the system role description
    
    messages_c=[
        {
            "role": "assistant",
            "content": [{ 
                            "type":"text",
                            "text": """Task: You are a video reviewer evaluating a sequence of actions presented as seven consecutive image uploads, which together represent a single video. You are going to accept the video if it completes the task and the video is consistent without glitches.
                       
                                    Query/Inference Phase:
                                        Inputs Provided:

                                            Textual Prompt: Describes the task the video should accomplish.
                                            Conditioning Image: Sets the fixed aspects of the scene.
                                            Sequence of Images (7 Frames): Represents consecutive moments in the video to be evaluated.

                                        Evaluation Process:

                                            View and Analyze Each Frame: Examine each of the seven images in sequence to understand the progression and continuity of actions.
                                            Assess Overall Coherence: Consider the sequence as a continuous scene to determine if the actions smoothly transition from one image to the next, maintaining logical progression.
                                            Check for Physical Accuracy: Ensure each frame adheres to the laws of physics, looking for any discrepancies in movement or positioning.
                                            Verify Task Completion: Check if the sequence collectively accomplishes the task described in the textual prompt.
                                            Identify Inconsistencies: Look for inconsistencies in object movement or overlaps that do not match the fixed scene elements shown in the conditioning image.

                                        Evaluation Criteria:

                                            Accept the sequence if it is as a coherent video which completes the task.
                                            Reject the sequence if any frame fails to meet the criteria, showing inconsistencies or not achieving the task. Reject even if there are the slightest errors. Be very strict.
                                            

                                        Response Requirement:

                                            Provide a single-word answer: Accept or Reject. Do not give reasoning.

                                        Additional Notes:

                                            You cannot request further clarification.
                                            The elements from the conditioning image must match those in each frame of the sequence."""
                            },
                     
            ]
        }
    ]
    responses = []
    print("Querying started")

    # Loop through each example provided
    for image_path, gif_path, text_prompt in zip(image_paths, gif_paths, text_prompts):
        # Conducting the conversation for each set of inputs
        if image_path == None or gif_path == None or text_prompt == None: 
            responses.append(None)
            continue
        while True:
            try:
                client = OpenAI(api_key=api_key)
                base_image = encode_image(image_path)
                base64_image_out = encode_gif(gif_path)
                messages_q=[
                    
                    
                {
                        "role": "user", 
                        "content": {
                                    "type":"image_url", 
                                    "image_url":  {
                                        "url": f"data:image/gif;base64,{base_image}"
                                        },
                                    }
                        },
                    
                    {
                        "role": "user", 
                        "content": [
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[0]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[1]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[2]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[3]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[4]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[5]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[6]}"
                                            },
                                        },
                            ]
                        },
                        {"role": "user", "content":"the conditioning image is the first upload, the next seven uploads are the key frames of the gif and the textual prompt is:" + text_prompt+ ". Return only the final decision"},
                        #,{"role": "user", "content": "also are there any collisions?"},
                ]
                response = client.chat.completions.create(model="gpt-4o",messages=messages_c + messages_q)


                # Store and print feedback
                feedback = response.choices[0].message.content
                responses.append(feedback)
                print("Feedback received:", feedback)
                break
            except:
                print("Bad Request, retrying")
                traceback.print_exc()
                continue
    print("Print feedback generated successfully")
    return responses

if __name__ == "__main__":
    # Example API Key and paths
    
    image_paths = ['feedback_fewshot/assembly_24_test.png']
    gif_paths = ['feedback_fewshot/assembly_checkpoint_276_out_3.gif']
    text_prompts = ['assembly']
    feedback_responses = chat_with_openai_rf(image_paths, gif_paths, text_prompts)
    print("All responses:", feedback_responses)

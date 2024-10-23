from openai import OpenAI
from PIL import Image
import os
import base64
import json
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

def chat_with_openai(image_paths, gif_paths, text_prompts):
    api_key = '' ##add_your_openai_api_key
    base64_image8 = encode_image("feedback_fewshot/assembly_24_test.png")
    base64_image8_out = encode_gif("feedback_fewshot/assembly_checkpoint_276_out_3.gif")
    #print(len(base64_image8_out))
    base64_image7 = encode_image("feedback_fewshot/assembly.png")
    base64_image7_out = encode_gif("feedback_fewshot/assembly_out_1iter_1.gif")
    base64_image6_out = encode_gif("feedback_fewshot/basketball_out_1iter_00.gif")
    base64_image6 = base64_image6_out[0]
    base64_image5_out = encode_gif("feedback_fewshot/hammer_checkpoint_310_out_0iter_7.gif")
    base64_image5 = base64_image5_out[0]
    # Initialize a conversation with the system role description
    
    messages_c=[
        {
            "role": "user",
            "content": [{ 
                            "type":"text",
                            "text": """
                            Task Description:
                                You are a video reviewer tasked with evaluating a series of actions depicted through eight consecutive image uploads. These images together simulate a video. This task is structured as a few-shot learning exercise, where you will first review three examples and then apply learned principles to new queries.

                            Learning Phase:
                                Example 1:
                                    Textual Prompt: assembly.
                                    Conditioning Image: """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8}"
                                },
                            },
                        { 
                            "type":"text",
                            "text": """
                                    GIF/Sequence of images:  """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[0]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[1]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[2]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[3]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[4]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[5]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image8_out[6]}"
                                },
                            },
                        
                        {
                            "type":"text", 
                            "text": " Ideal Feedback : Task Incomplete. Move the manipulator along with the ring more forward and to the left."
                            },
                        { 
                            "type":"text",
                            "text": """
                                Example 2:
                                    Textual Prompt: assembly.
                                    Conditioning Image: """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7}"
                                },
                            },
                        { 
                            "type":"text",
                            "text": """
                                    GIF/Sequence of images:  """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[0]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[1]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[2]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[3]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[4]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[5]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[6]}"
                                },
                            },
                        {
                            "type":"text", 
                            "text": " Ideal Feedback : Inconsistent glitch. Ring jumps without any force acting on it. Move the manipulator to the ring use it to pick ring up"
                            },
                        
                        { 
                            "type":"text",
                            "text": """
                                Example 3:
                                    Textual Prompt:hammer.
                                    Conditioning Image: """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5}"
                                },
                            },
                        { 
                            "type":"text",
                            "text": """
                                    GIF/Sequence of images:  """
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[0]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[1]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[2]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[3]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[4]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[5]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[6]}"
                                },
                            },
                        {
                            "type": "text", 
                            "text": "Ideal Feedback : The hammer must push the nail into the wood. Maintain consistency by ensuring the hammer and nail do not merge."
                            },
                        {
                            "type": "text",
                            "text": """ 
                                    Query/Inference Phase:
                                        Inputs Provided:

                                            Textual Prompt: Describes the intended outcome or task the video aims to accomplish.
                                            Conditioning Image: Establishes the fixed elements of the scene.
                                            Sequence of Images (7 Frames): Illustrates consecutive moments in the video, representing the action sequence.

                                        Evaluation Process:

                                            Frame-by-Frame Analysis: Carefully examine each of the seven images to understand the progression and continuity of actions.
                                            Assess Overall Coherence: Evaluate the sequence as a whole to determine if the actions transition smoothly from one frame to the next while maintaining logical progression.
                                            Physical Accuracy Check: Ensure each frame complies with the laws of physics, identifying any discrepancies in movement or positioning.
                                            Verify Task Completion: Confirm if the sequence as a whole accomplishes the task described in the textual prompt. Accept if the final frame is close to the task goal.
                                            Identify Inconsistencies: Detect inconsistencies in object movement or overlaps that contradict the fixed scene elements depicted in the conditioning image.

                                        Evaluation Criteria:

                                            Feedback: Based on your evaluation, provide a concise, constructive sentence suggesting specific improvements. Focus on enhancing physical accuracy and task fulfillment based on identified inconsistencies or discrepancies.

                                        Response Requirement:

                                            Feedback must be derived from your observations during the evaluation and not exceed 20 words.

                                        Additional Notes:

                                            Further clarification cannot be requested.
                                            The elements from the conditioning image must match those in each frame of the sequence."""
                            }
                        
            ]
        }
    ]
    tasks_dict = {
    "door-open": "The robot arm has to open the door by using the door handle.",
    "door-close": "The robot arm has to close the door by pushing the door or the handle.",
    "basketball": "The robot arm has to pick up the basketball and take it above the hoop.",
    "shelf-place": "The robot arm has to pick up the blue cube and take it near the loaction marked on the shelf.",
    "button-press": "The robot arm has to press the red button from the side.",
    "button-press-topdown": "The robot arm has to press the red button from the top.",
    "faucet-close": "The robot arm has to use the red faucet handle and turn it anti-clockwise.",
    "faucet-open": "The robot arm has to use the red faucet handle and turn it clockwise.",
    "handle-press": "The robot arm has to press the red handle downward.",
    "hammer": "The robot arm has to grip and pick up the hammer with a red handle and hit the peg on the box.",
    "assembly": "The robot arm has to pick up the ring and place it into the red peg."
}
    responses = []

    # Loop through each example provided
    for image_path, gif_path, text_prompt in zip(image_paths, gif_paths, text_prompts):
        # Conducting the conversation for each set of inputs
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
                        {"role": "user", "content":"the conditioning image is the first upload, the next seven uploads are the key frames of the gif and the textual prompt is:" + text_prompt +". The final goal task to check is:" + tasks_dict[text_prompt] + ".Return only final feedback"},
                ]
                response = client.chat.completions.create(model="gpt-4o-mini",messages=messages_c + messages_q)


                # Store and print feedback
                feedback = response.choices[0].message.content
                responses.append(feedback)
                print("Feedback received:", feedback)
                break
            except:
                print("Bad request, retrying")
                traceback.print_exc()
                continue

    return responses

if __name__ == "__main__":
    # Example API Key and paths
    
    
    image_paths = ['feedback_fewshot/assembly_24_test.png']
    gif_paths = ['feedback_fewshot/assembly_checkpoint_276_out_3.gif']
    text_prompts = ['assembly']
    
    feedback_responses = chat_with_openai(image_paths, gif_paths, text_prompts)

    with open("feedback_suggestive_mw.json", "w") as final:
        json.dump(feedback_responses, final)

    print("All responses:", feedback_responses)

from openai import OpenAI
from PIL import Image
import os
import base64

api_key = 'sk-proj-949BygC19E83huSvkr4xT3BlbkFJwS60lvFDMhK4A2wTFeFM'
client = OpenAI(api_key=api_key)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_gif(image_path):
    frame = Image.open(image_path)
    nframes = 0
    encoded = []
    while frame:
        image_file = os.path.basename(image_path) + "-" + ".jpg"
        frame.convert('RGB').save(image_file)
        with open(image_file, "rb") as image:
            encoded.append(base64.b64encode(image.read()).decode('utf-8'))
        nframes += 1
        if nframes == 8: break
    
    return encoded

def chat_with_openai(api_key, image_paths, gif_paths, text_prompts):
    base64_image7 = encode_image("examples/im_7.jpg")
    base64_image7_out = encode_gif("examples/im_7_out.gif")
    base64_image6 = encode_image("examples/im_6.jpg")
    base64_image6_out = encode_gif("examples/im_6_out.gif")
    base64_image5 = encode_image("examples/im_5.jpg")
    base64_image5_out = encode_gif("examples/im_5_out.gif")
    # Initialize a conversation with the system role description
    
    messages_c=[
        {
            "role": "assistant",
            "content": [{ 
                            "type":"text",
                            "text": "You are a video generation reviewer. You are given a textual prompt, an conditioning image and a video as a gif . The gif is given as 8 key frames. You are going to give feedback in the form of a sentence in a short, concise manner and not more then 20 words. The feedback has to be in the form of suggestions which improve how correct the video is in terms of  whether it follows physics and whether it accomplishes the task description given in the text prompt. Also evaluate whether there any collisions, if yes mention it in you feedback.  You will take in the input from the user. Keep in mind that you will have to explain how to solve the issue not just state it. The aspects of the scene are fixed to what is given in the conditioning image and those will not be different in the gif. I will give you three examples of feedback. Use these three examples to learn how to give better feedback to user queries.  First I will give you three examples. "
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7}"
                                },
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
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image7_out[7]}"
                                },
                            },
                        {
                            "type":"text", 
                            "text": "First example : the conditioning image is the first upload, the next eight uploads are the key frames of the gif and the textual prompt is : move the blue fork to the right bottom burner. Feedback : Maintain continuity of the manipulator arm and start from the correct orientation of the arm"
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[0]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[1]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[2]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[3]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[4]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[5]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[6]}"
                                },
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image6_out[7]}"
                                },
                            },
                        {
                            "type":"text", 
                            "text": "Second Example : the conditioning image is the first upload, the next eight uploads are the key frames of the gif and the textual prompt is: moved the blue object inside the silver pot. Feedback : Pick the blue fork, not the green object and complete the trajectory"
                            },
                        {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5}"
                                },
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
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/jpeg;base64,{base64_image5_out[7]}"
                                },
                            },
                        {
                            "type": "text", 
                            "text": "Last example : the conditioning image is the first upload, the next eight uploads are the key frames of the gif and the textual prompt is: put the blue fork on the bottom right burner. Feedback : Pick the blue fork not the green object. Place it on the bottom right with the knobs as reference"
                            },
                        {
                            "type": "text",
                            "text": "You will take in the query input from the user, and give feedback using the examples as reference, keeping in mind that the feedback should comment on following physics(such as collision) and whether the task description given in the text prompt is accomplished."
                            }
            ]
        }
    ]
    responses = []

    # Loop through each example provided
    for image_path, gif_path, text_prompt in zip(image_paths, gif_paths, text_prompts):
        # Conducting the conversation for each set of inputs
        base_image = encode_image(image_path)
        base64_image_out = encode_gif(gif_path)
        messages_q=[
            
            #{"role": "assistant", "content": "Please upload the conditioning image."},
            {
                "role": "user", 
                "content": {
                            "type":"image_url", 
                            "image_url":  {
                                "url": f"data:image/gif;base64,{base_image}"
                                },
                            }
                },
            #{"role": "assistant", "content": "Now, upload the corresponding gif as 8 key frames."},
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
                            {
                                "type":"image_url", 
                                "image_url":  {
                                    "url": f"data:image/jpeg;base64,{base64_image_out[7]}"
                                    },
                                },
                    ]
                },
                {"role": "user", "content":"the conditioning image is the first upload, the next eight uploads are the key frames of the gif and the textual prompt is:" + text_prompt},
                #,{"role": "user", "content": "also are there any collisions?"},
        ]
        response = client.chat.completions.create(model="gpt-4o",messages=messages_c + messages_q)


        # Store and print feedback
        feedback = response.choices[0].message.content
        responses.append(feedback)
        print("Feedback received:", feedback)

    return responses

if __name__ == "__main__":
    # Example API Key and paths
    
    image_paths = ["examples/im_2.jpg", "examples/im_4.jpg"]
    gif_paths = ["examples/im_2_out.gif", "examples/im_4_out.gif"]
    text_prompts = [
        "take the blue spoon off the burner and put it on the right side of the burner",
        "take the red object out of the pot and put it on the left burner"
    ]

    # Call the function with example data
    feedback_responses = chat_with_openai(api_key, image_paths, gif_paths, text_prompts)
    print("All responses:", feedback_responses)

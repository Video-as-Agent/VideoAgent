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

# def encode_gif(image_path):
#     frame = Image.open(image_path)
#     nframes = 0
#     encoded = []
#     while frame:
#         image_file = "feedback_fewshot/.cache/"+os.path.basename(image_path) + "-" +".jpg"
#         frame.convert('RGB').save(image_file)
#         with open(image_file, "rb") as image:
#             encoded.append(base64.b64encode(image.read()).decode('utf-8'))
#         nframes += 1
#         if nframes == 8: break
#         frame.seek(nframes)
#         # print(nframes)
    
#     return encoded

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
    api_key = 'sk-proj-949BygC19E83huSvkr4xT3BlbkFJwS60lvFDMhK4A2wTFeFM'
    base64_image8 = encode_image("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/assembly_24_test.png")
    base64_image8_out = encode_gif("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/assembly_checkpoint_276_out_3.gif")
    # print(len(base64_image8_out))
    base64_image7 = encode_image("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/assembly.png")
    base64_image7_out = encode_gif("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/assembly_out_1iter_1.gif")
    base64_image6_out = encode_gif("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/basketball_out_1iter_00.gif")
    base64_image6 = base64_image6_out[0]
    base64_image5_out = encode_gif("/home/ubuntu/achint/SI-GenSim/flowdiffusion/feedback_fewshot/hammer_checkpoint_310_out_0iter_7.gif")
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
                                    # {
                                    #     "type":"image_url", 
                                    #     "image_url":  {
                                    #         "url": f"data:image/jpeg;base64,{base64_image_out[7]}"
                                    #         },
                                    #     },
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
                # frame = Image.open(gif_path)
                # nframes = 0
                # try:
                #     while frame:
                #         nframes = nframes+1
                # except EOFError:
                #     pass
                # print("len, " , nframes)
                traceback.print_exc()
                continue
    print("Print feedback generated successfully")
    return responses

if __name__ == "__main__":
    # Example API Key and paths
    
    image_paths = ['feedback_fewshot/assembly_24_test.png']#, 'AVDC_gen/AVDC/data/img_test/door-open_1.png', 'AVDC_gen/AVDC/data/img_test/door-open_2.png', 'AVDC_gen/AVDC/data/img_test/drawer-open_0.png', 'AVDC_gen/AVDC/data/img_test/drawer-open_1.png', 'AVDC_gen/AVDC/data/img_test/drawer-open_2.png', 'AVDC_gen/AVDC/data/img_test/door-close_0.png', 'AVDC_gen/AVDC/data/img_test/door-close_1.png', 'AVDC_gen/AVDC/data/img_test/door-close_2.png', 'AVDC_gen/AVDC/data/img_test/basketball_0.png', 'AVDC_gen/AVDC/data/img_test/basketball_1.png', 'AVDC_gen/AVDC/data/img_test/basketball_2.png', 'AVDC_gen/AVDC/data/img_test/shelf-place_0.png', 'AVDC_gen/AVDC/data/img_test/shelf-place_1.png', 'AVDC_gen/AVDC/data/img_test/shelf-place_2.png', 'AVDC_gen/AVDC/data/img_test/button-press_0.png', 'AVDC_gen/AVDC/data/img_test/button-press_1.png', 'AVDC_gen/AVDC/data/img_test/button-press_2.png', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_0.png', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_1.png', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_2.png', 'AVDC_gen/AVDC/data/img_test/faucet-close_0.png', 'AVDC_gen/AVDC/data/img_test/faucet-close_1.png', 'AVDC_gen/AVDC/data/img_test/faucet-close_2.png', 'AVDC_gen/AVDC/data/img_test/faucet-open_0.png', 'AVDC_gen/AVDC/data/img_test/faucet-open_1.png', 'AVDC_gen/AVDC/data/img_test/faucet-open_2.png', 'AVDC_gen/AVDC/data/img_test/handle-press_0.png', 'AVDC_gen/AVDC/data/img_test/handle-press_1.png', 'AVDC_gen/AVDC/data/img_test/handle-press_2.png', 'AVDC_gen/AVDC/data/img_test/hammer_0.png', 'AVDC_gen/AVDC/data/img_test/hammer_1.png', 'AVDC_gen/AVDC/data/img_test/hammer_2.png', 'AVDC_gen/AVDC/data/img_test/assembly_0.png', 'AVDC_gen/AVDC/data/img_test/assembly_1.png', 'AVDC_gen/AVDC/data/img_test/assembly_2.png']
    gif_paths = ['feedback_fewshot/assembly_checkpoint_276_out_3.gif']#, 'AVDC_gen/AVDC/data/img_test/door-open_1_out.gif', 'AVDC_gen/AVDC/data/img_test/door-open_2_out.gif', 'AVDC_gen/AVDC/data/img_test/drawer-open_0_out.gif', 'AVDC_gen/AVDC/data/img_test/drawer-open_1_out.gif', 'AVDC_gen/AVDC/data/img_test/drawer-open_2_out.gif', 'AVDC_gen/AVDC/data/img_test/door-close_0_out.gif', 'AVDC_gen/AVDC/data/img_test/door-close_1_out.gif', 'AVDC_gen/AVDC/data/img_test/door-close_2_out.gif', 'AVDC_gen/AVDC/data/img_test/basketball_0_out.gif', 'AVDC_gen/AVDC/data/img_test/basketball_1_out.gif', 'AVDC_gen/AVDC/data/img_test/basketball_2_out.gif', 'AVDC_gen/AVDC/data/img_test/shelf-place_0_out.gif', 'AVDC_gen/AVDC/data/img_test/shelf-place_1_out.gif', 'AVDC_gen/AVDC/data/img_test/shelf-place_2_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press_0_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press_1_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press_2_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_0_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_1_out.gif', 'AVDC_gen/AVDC/data/img_test/button-press-topdown_2_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-close_0_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-close_1_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-close_2_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-open_0_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-open_1_out.gif', 'AVDC_gen/AVDC/data/img_test/faucet-open_2_out.gif', 'AVDC_gen/AVDC/data/img_test/handle-press_0_out.gif', 'AVDC_gen/AVDC/data/img_test/handle-press_1_out.gif', 'AVDC_gen/AVDC/data/img_test/handle-press_2_out.gif', 'AVDC_gen/AVDC/data/img_test/hammer_0_out.gif', 'AVDC_gen/AVDC/data/img_test/hammer_1_out.gif', 'AVDC_gen/AVDC/data/img_test/hammer_2_out.gif', 'AVDC_gen/AVDC/data/img_test/assembly_0_out.gif', 'AVDC_gen/AVDC/data/img_test/assembly_1_out.gif', 'AVDC_gen/AVDC/data/img_test/assembly_2_out.gif']
    text_prompts = ['assembly']#, 'door-open', 'door-open', 'drawer-open', 'drawer-open', 'drawer-open', 'door-close', 'door-close', 'door-close', 'basketball', 'basketball', 'basketball', 'shelf-place', 'shelf-place', 'shelf-place', 'button-press', 'button-press', 'button-press', 'button-press-topdown', 'button-press-topdown', 'button-press-topdown', 'faucet-close', 'faucet-close', 'faucet-close', 'faucet-open', 'faucet-open', 'faucet-open', 'handle-press', 'handle-press', 'handle-press', 'hammer', 'hammer', 'hammer', 'assembly', 'assembly', 'assembly']
    
    #image_paths = ['AVDC_gen/AVDC/examples/assembly.png','AVDC_gen/AVDC/data/iterative_test/assembly_24_test.png','AVDC_gen/AVDC/data/iterative_test/assembly_24_test.png','AVDC_gen/AVDC/data/iterative_test/assembly_24_test.png','AVDC_gen/AVDC/data/iterative_test/assembly_24_test.png']
    #gif_paths = ['AVDC_gen/AVDC/examples/assembly_out.gif','AVDC_gen/AVDC/data/iterative_test/assembly_checkpoint_276_out_1.gif', 'AVDC_gen/AVDC/data/iterative_test/assembly_checkpoint_276_out_2.gif', 'AVDC_gen/AVDC/data/iterative_test/assembly_checkpoint_276_out_3.gif','AVDC_gen/AVDC/data/iterative_test/assembly_checkpoint_276_out_4.gif']
    #text_prompts = ['assembly','assembly','assembly','assembly','assembly']
    # Call the function with example data
    feedback_responses = chat_with_openai_rf(image_paths, gif_paths, text_prompts)
    # with open("training_feedback_mw.json", "w") as final:
    #    json.dump(feedback_responses, final)
    print("All responses:", feedback_responses)

"""
 #         Learning Phase:
                        #         Example 1:
                        #             Textual Prompt: assembly.
                        #             Conditioning Image: """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8}"
                        #         },
                        #     },
                        # { 
                        #     "type":"text",
                        #     "text": """
                        #             GIF/Sequence of images:  """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[0]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[1]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[2]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[3]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[4]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[5]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[6]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image8_out[7]}"
                        #         },
                        #     },
                        
                        # {
                        #     "type":"text", 
                        #     "text": " Ideal Feedback : Reject. Reason: The ring should be placed into the cylinder"
                        #     },
                        # { 
                        #     "type":"text",
                        #     "text": """
                        #         Example 2:
                        #             Textual Prompt: basketball.
                        #             Conditioning Image: """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6}"
                        #         },
                        #     },
                        # { 
                        #     "type":"text",
                        #     "text": """
                        #             GIF/Sequence of images:  """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[0]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[1]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[2]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[3]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[4]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[5]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[6]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image6_out[7]}"
                        #         },
                        #     },
                        # {
                        #     "type":"text", 
                        #     "text": " Ideal Feedback : Accept Reason : Task completed and good quality video"
                        #     },
                        # { 
                        #     "type":"text",
                        #     "text": """
                        #         Example 3:
                        #             Textual Prompt:hammer.
                        #             Conditioning Image: """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5}"
                        #         },
                        #     },
                        # { 
                        #     "type":"text",
                        #     "text": """
                        #             GIF/Sequence of images:  """
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[0]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[1]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[2]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[3]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[4]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[5]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[6]}"
                        #         },
                        #     },
                        # {
                        #     "type":"image_url", 
                        #     "image_url":  {
                        #         "url": f"data:image/jpeg;base64,{base64_image5_out[7]}"
                        #         },
                        #     },
                        # {
                        #     "type": "text", 
                        #     "text": "Ideal Feedback : Reject. Reason: hammer and nail merge."
                        #     },
                        # {
                        #     "type": "text",
                        #     "text": """ 
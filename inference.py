from openai import OpenAI
import pandas as pd
from pathlib import Path
import argparse
from typing import Literal
import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from prompts import enhanced_prompt, physical_law_prompt, mismatch_prompt
from tarsier.tasks.utils import load_model_and_processor
from tarsier.dataset.utils import *
import os
from tqdm import tqdm
import subprocess

def sa_score_generation(round, video_dir):
    prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: Does this video entail the description: <prompt>?
    AI: """
    prompt_column = f"prompt_{round}"

    df = pd.read_csv("data_df.csv")
    for i in range(len(df)):
        df.loc[i, "caption"] = prompt.replace("<prompt>", df.loc[i, prompt_column])

    video_dir = Path(video_dir)
    for video_path in video_dir.glob(f"*{round}.mp4"):
        df["videopath"] = video_path
    df = df[["videopath", "caption"]]
    df.to_csv(f"eval_csv/sa_round{round}.csv")

    # Command to run
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python", "videophy/videocon/training/pipeline_video/entailment_inference.py",
        "--input_csv", f"eval_csv/sa_round{round}.csv",
        "--output_csv", f"eval_csv/sa_result_round{round}.csv",
        "--checkpoint", "videophy/videocon_physics"
    ]

    # Run the command
    subprocess.run(" ".join(command), shell=True)

    result_df = pd.read_csv(f"eval_csv/sa_result_round{round}.csv", header=None)
    return result_df.iloc[0, -1]


def pc_score_generation(round, video_dir):

    df = pd.read_csv("data_df.csv")
    prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <|video|>
    Human: Does this video follow the physical laws?
    AI: """

    video_dir = Path(video_dir)
    for video_path in video_dir.glob(f"*{round}.mp4"):
        df["videopath"] = video_path
        df["caption"] = prompt
    df = df[["videopath", "caption"]]
    df.to_csv(f"eval_csv/pc_round{round}.csv")

    # Command to run
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python", "videophy/videocon/training/pipeline_video/entailment_inference.py",
        "--input_csv", f"eval_csv/pc_round{round}.csv",
        "--output_csv", f"eval_csv/pc_result_round{round}.csv",
        "--checkpoint", "videophy/videocon_physics"
    ]

    # Run the command
    subprocess.run(" ".join(command), shell=True)

    result_df = pd.read_csv(f"eval_csv/pc_result_round{round}.csv", header=None)
    return result_df.iloc[0, -1]
                        


def process_one(model, processor, prompt, video_file, generate_kwargs):
    inputs = processor(prompt, video_file, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        print(f"Prompt: {inputs.pop('prompt')}")
    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text

def caption_generation(main_objects, video_path):
    instruction = "This is a video of <>, give extremely detailed description on the motion and the deformation performed in the video, what physics law does it obey and disobey in this process?".replace("<>", main_objects)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="tarsier/Tarsier-34b", type=str)
    parser.add_argument('--instruction', type=str, default=instruction, help='Input prompt.')
    parser.add_argument('--input_path', type=str, default=video_path, help='Path to video/image; or Dir to videos/images')
    parser.add_argument("--max_n_frames", type=int, default=8, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")
    parser.add_argument("--output_file", type=str, default="tarsier/output.txt", help="Set output file path.")
    parser.add_argument("--round_num", required=True, default=2, help="prompt refinement round")
    parser.add_argument("--gpt_api", type=str, required=True, default=2, help="prompt refinement round")

    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_name_or_path, max_n_frames=args.max_n_frames)
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True
    }
    assert os.path.exists(args.input_path), f"input_path not exist: {args.input_path}"
    if os.path.isdir(args.input_path):
        input_files = [os.path.join(args.input_path, fn) for fn in os.listdir(args.input_path) if get_visual_type(fn) in ['video', 'gif', 'image']]
    elif get_visual_type(args.input_path) in ['video', 'gif', 'image']:
        input_files = [args.input_path]
    assert len(input_files) > 0, f"None valid input file in: {args.input_path} {VALID_DATA_FORMAT_STRING}"

    for input_file in tqdm(input_files, desc="Generating..."):
        visual_type = get_visual_type(input_file)
        if args.instruction:
            prompt = args.instruction
            prompt = "<video>\n" + prompt.replace("<image>", "").replace("<video>", "")
        else:
            if visual_type == 'image':
                prompt = "<image>\nDescribe the image in detail."
            else:
                prompt = "<video>\nDescribe the video in detail."
        
        pred = process_one(model, processor, prompt, input_file, generate_kwargs)
        with open(args.output_file, "w") as f:
            f.write(pred)
        print(f"Prediction: {pred}")
        print('-'*100)
    return pred


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.to("cuda")

    pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    if generate_type == "i2v":
        video_generate = pipe(
            prompt=prompt,
            image=image,  # The path of the image to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    export_to_video(video_generate, output_path, fps=8)

def video_generation(prompt_path, output_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default=prompt, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default=output_path, help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--round_num", required=True, default=2, help="prompt refinement round")
    parser.add_argument("--gpt_api", type=str, required=True, default=2, help="prompt refinement round")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Physical-grounded video generation with PhyT2V")
    parser.add_argument("--round_num", type=int, required=True, default=2, help="prompt refinement round number")
    parser.add_argument("--gpt_api", type=str, required=True, help="chatgpt api key")
    args = parser.parse_args()

    PROMPT_PATH = Path("prompt.txt")
    client = OpenAI(
      api_key=args.gpt_api
    )

    with open(PROMPT_PATH) as f:
        prompt = f.read()
    data_df = pd.DataFrame([{"prompt_1": prompt}])

    print("-"*30+"Physcial law and main object extraction"+"-"*30)
    main_objects, physical_law = physical_law_prompt(client, prompt)
    data_df["main_object"] = main_objects
    data_df["physical_law"] = physical_law
    video_dir = "output_videos" 
    if not Path(video_dir).exists():
        Path(video_dir).mkdir()
    eval_dir = "eval_csv"
    if not Path("eval_csv").exists():
        Path(eval_dir).mkdir()
        
    print("-"*30+"Round 1 video generation"+"-"*30)
    video_generation(PROMPT_PATH, video_dir + f"/output1.mp4")

    for i in range(1, args.round_num+1):
      caption_column = f"caption_{i}"
      mismatch_column = f"mismatch_{i}"
      sa_column = f"sa_{i}"
      pc_column = f"pc_{i}"
      prompt_column = f"prompt_{i+1}"
      video_path = video_dir + f"/output{i}.mp4"
      video_output_path = video_dir + f"/output{i+1}.mp4"

      print("-"*30+f"Round {i} caption generation"+"-"*30)
      video_caption = caption_generation(main_objects, video_path)
      data_df[caption_column] = video_caption

      print("-"*30+f"Round {i} mismatch generation"+"-"*30)
      mismatch = mismatch_prompt(client, prompt, video_caption)
      data_df[mismatch_column] = mismatch

      data_df.to_csv("data_df.csv", index_label="index")
      
      print("-"*30+f"Round {i} SA&PC score generation"+"-"*30)
      sa_score = sa_score_generation(i, video_dir)
      pc_score = pc_score_generation(i, video_dir)
      data_df[sa_column] = float(sa_score)
      data_df[pc_column] = float(pc_score)

      print("-"*30+f"Round {i+1} refined prompt generation"+"-"*30)
      score = float(sa_score)*0.5 + float(pc_score)*0.5
      refined_prompt = enhanced_prompt(client, prompt, physical_law, mismatch, str(score))
      with open(PROMPT_PATH, "w") as f:
          f.write(refined_prompt)

      print("-"*30+f"Round {i+1} video generation"+"-"*30)
      video_generation(PROMPT_PATH, video_output_path)
      data_df[prompt_column] = prompt

      data_df.to_csv("data_df.csv", index_label="index")

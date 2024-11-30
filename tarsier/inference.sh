MODEL_NAME_OR_PATH="Tarsier-34b"
VIDEO_FILE="../output.mp4"

python3 -m tasks.inference_quick_start \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --instruction "This is a video of cream and soup, give extremely detailed description on the motion and the deformation performed in the video, what physics law does it obey and disobey in this process?" \
  --input_path $VIDEO_FILE
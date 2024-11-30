import os
import csv
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type = str, required = True, help = 'input csv file')
parser.add_argument('--output_file', type = str, help = 'output csv file')
parser.add_argument('--pretrained_ckpt', type = str, required = True, help = 'pretrained ckpt')
parser.add_argument('--trained_ckpt', type = str, help = 'trained ckpt')
parser.add_argument('--lora_r', type = int, default = 32)
parser.add_argument('--use_lora', action = 'store_true', help = 'lora model')
parser.add_argument('--all_params', action = 'store_true', help = 'all params')
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--num_frames', type = int, default = 32)

args = parser.parse_args()

PROMPT_FEEDBACK = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: What is the misalignment between this video and the description: "{caption}"?
AI: '''

generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

class VideoCaptionDataset(Dataset):

    def __init__(self, input_file):
        self.data = pd.read_csv(input_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = {}
        item['videopath'] = self.data.iloc[index]['videopath']
        item['neg_caption'] = self.data.iloc[index]['neg_caption']
        return item

def get_nle(args, model, processor, tokenizer, dataloader):

    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            videopaths = batch['videopath']
            neg_caption = batch['neg_caption'][0]
            prompts = [PROMPT_FEEDBACK.format(caption = neg_caption)] 
            inputs = processor(text=prompts, videos=videopaths, num_frames=args.num_frames, return_tensors='pt')
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            res = model.generate(**inputs, **generate_kwargs)
            generated_nle = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

            with open(args.output_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([videopaths[0], neg_caption, generated_nle])

def main():

    # Create dataloader
    dataset = VideoCaptionDataset(args.input_file)
    dataloader = DataLoader(dataset, batch_size = args.batch_size)

    pretrained_ckpt = args.pretrained_ckpt

    # Processors
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_ckpt)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={'':0}
    )

    if args.use_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
        if args.all_params:
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
                inference_mode=True, 
                r=args.lora_r, 
                lora_alpha=32, 
                lora_dropout=0.05
            )
        else:
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj)', 
                inference_mode=True, 
                r=args.lora_r, 
                lora_alpha=32, 
                lora_dropout=0.05
            )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        with open(args.trained_ckpt, 'rb') as f:
            ckpt = torch.load(f, map_location = torch.device(f"cuda:0"))
        model.load_state_dict(ckpt)
        model = model.to(torch.bfloat16)
        print('Model Loaded')
        
    model.eval()

    # get nle
    get_nle(args, model, processor, tokenizer, dataloader)



if __name__  == "__main__":
    main()
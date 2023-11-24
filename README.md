# README

Test Branch for testing separated captions for sdxl training.
FINETUNING ONLY! And only tested for the following workflow.
This branch will break other stuff!

## What can you do with this
1) create a metadata file with separated captions for text_encoder1 and text_encoder2
2) finetune a model with separated captions
3) before the training process an initial sample image will be created in order better understand the trainings process
4) separated captions for sample images

Info about the text encoders:
text_encoder1: CLIPTextModel - 'openai/clip-vit-large-patch14' / L / in Comfy CLIP_L
text_encoder2: CLIPTextModelWithProjection - 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' / G / in Comfy CLIP_G -> the "sdxl" decoder

So in the following notes everything like 
captions_g / captionsG / (G).. refers to the text_encoder2
captions_l / captionsL / (L).. refers to the text_encoder1

## Prepare your images and captions files
tested it with a single folder (like images)
01.jpg
01.caption -> file for (G) text_encoder2
01.txt -> file for (L) text_encoder1
.
.

(see notes at the bottom for some thoughts on how to caption the images)

use the following scripts to create the required metadatafile and prepare the bucket latents
sdxl_merge_captions_g_to_metadata.py
sdxl_merge_captions_l_to_metadata.py
prepare_buckets_latents.py

you can use it like this:

python .\finetune\sdxl_merge_captions_g_to_metadata.py X:\Path to training folder\images X:\Path to training folder\meta_cap.json

python .\finetune\sdxl_merge_captions_l_to_metadata.py X:\Path to training folder\images X:\Path to training folder\meta_cap.json

python .\finetune\prepare_buckets_latents.py X:\Path to training folder\images X:\Path to training folder\meta_cap.json X:\Path to training folder\meta_lat.json X:\Path to training folder\sd_xl_base_1.0.safetensors --batch_size 1 --max_resolution=1024,1024 --min_bucket_reso=1024 --max_bucket_reso=2048 --mixed_precision=bf16

you will get a metafile with following info:

```
{
  "filename": {
    "captionG": "caption for (G) text_encoder2",
    "captionL": "caption for (L) text_encoder1",
    "train_resolution": [
      896,
      1152
    ]
  },
}
```

## Training
after this you can call your training file (sample config I used for creating the samples)

```
accelerate launch 
--num_cpu_threads_per_process=4 
"./sdxl_train.py" 
--pretrained_model_name_or_path="X:\Path to training folder\sd_xl_base_1.0.safetensors" 
--in_json X:\Path to training folder\meta_lat.json
--train_data_dir="X:\Path to training folder\images" 
--resolution="1024,1024" 
--output_dir="X:\Path to training folder\output" 
--logging_dir="X:\Path to training folder\logs" 
--save_model_as=safetensors 
--full_bf16 
--output_name="finetuned_model" 
--max_data_loader_n_workers="0" 
--learning_rate="1e-5"
--train_batch_size="1" 
--max_train_steps="500" 
--mixed_precision="bf16" 
--save_precision="bf16" 
--cache_latents 
--cache_latents_to_disk 
--optimizer_type="Adafactor" 
--optimizer_args scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01 
--gradient_checkpointing 
--max_grad_norm=0.0 
--noise_offset=0.0357 
--adaptive_noise_scale=0.00357
--sample_prompts="X:\Path to training folder\prompt.txt"
--sample_every_n_steps="10"
--sample_sampler="euler_a"
--no_half_vae
```

## Captioning

For now I tested:
   (L) for style like "photographic", "anime"
   (G) for prompt like "woman, portrait, in nature"
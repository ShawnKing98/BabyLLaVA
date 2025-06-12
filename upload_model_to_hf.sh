source ~/.bashrc
conda activate llava

huggingface-cli upload wsashawn/llava_lora_ft_SAYCam_gt0.2_bs640_lr2e-4 ./checkpoints/llava_lora_ft_SAYCam_gt0.2_bs640_lr2e-4
huggingface-cli upload wsashawn/llava_lora_ft_SAYCam_gt0.2_bs320_lr6e-5 ./checkpoints/llava_lora_ft_SAYCam_gt0.2_bs320_lr6e-5
huggingface-cli upload wsashawn/llava_lora_ft_SAYCam_gt0.2_bs320_lr6e-4 ./checkpoints/llava_lora_ft_SAYCam_gt0.2_bs320_lr6e-4
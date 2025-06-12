cd /projectnb/ivc-ml/wsashawn/LLaVA

# finetune
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr6e-4 scc_finetune_babylm.sh --lr 6e-4 --gacc 2 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 2 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr6e-5 scc_finetune_babylm.sh --lr 6e-5 --gacc 2 --bs 40

# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*1*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 1 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*4*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 4 --bs 40

# phase 1
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*1*4))_lr1e-3 scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 80
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*1*4))_lr3e-3 scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 80
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*1*4))_lr3e-4 scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 80

# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((40*1*4))_lr1e-3 scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 40
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((40*1*4))_lr3e-3 scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 40
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((40*1*4))_lr3e-4 scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 40

# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*2*4))_lr1e-3 scc_phase1_babylm.sh --lr 1e-3 --gacc 2 --bs 80
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*2*4))_lr3e-3 scc_phase1_babylm.sh --lr 3e-3 --gacc 2 --bs 80
# qsub -N babygpt_llava_p1_SAYCam_gt0.2_bs$((80*2*4))_lr3e-4 scc_phase1_babylm.sh --lr 3e-4 --gacc 2 --bs 80

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 64
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 64
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 64

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 32
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 32
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 32

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 128
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 128
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 128

# bash scc_phase2_babylm_2.sh --lr 2e-5 --gacc 1 --bs 32 --ep 10
# bash scc_phase2_babylm_2.sh --lr 2e-6 --gacc 1 --bs 32
# bash scc_phase2_babylm_2.sh --lr 2e-4 --gacc 1 --bs 32

# bash scc_phase2_babylm_2.sh --lr 2e-5 --gacc 1 --bs 16
# bash scc_phase2_babylm_2.sh --lr 2e-5 --gacc 1 --bs 32 --ep 20

# bash scc_phase1_babylm_distill_2.sh --lr 1e-3 --gacc 1 --bs 80
# bash scc_phase1_babylm_distill_2.sh --lr 3e-3 --gacc 1 --bs 80
# bash scc_phase1_babylm_distill_2.sh --lr 3e-4 --gacc 1 --bs 80

# bash scc_phase1_babylm_distill_2.sh --lr 1e-3 --gacc 1 --bs 40
# bash scc_phase1_babylm_distill_2.sh --lr 3e-3 --gacc 1 --bs 40
# bash scc_phase1_babylm_distill_2.sh --lr 3e-4 --gacc 1 --bs 40

# bash scc_phase1_babylm_distill_2.sh --lr 1e-3 --gacc 2 --bs 80
# bash scc_phase1_babylm_distill_2.sh --lr 3e-3 --gacc 2 --bs 80
# bash scc_phase1_babylm_distill_2.sh --lr 3e-4 --gacc 2 --bs 80

learning_rates=("2e-3" "2e-4" "2e-5")
batch_sizes=(32 64 128)
for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        bash scc_phase2_babylm_distill_2.sh \
            --lr $lr \
            --gacc 1 \
            --bs $bs \
            --connector /projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/llava_babygpt_dino_phase1_only_random_data_75_bs160_lr1e-3_ep5/mm_projector.bin \
            --ep 5
    done
done
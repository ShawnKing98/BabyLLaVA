cd /projectnb/ivc-ml/wsashawn/LLaVA

# ======= finetune =========
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr6e-4 scc_finetune_babylm.sh --lr 6e-4 --gacc 2 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 2 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*2*4))_lr6e-5 scc_finetune_babylm.sh --lr 6e-5 --gacc 2 --bs 40

# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*1*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 1 --bs 40
# qsub -N ft_lora_SAYCam_gt0.2_bs$((40*4*4))_lr2e-4 scc_finetune_babylm.sh --lr 2e-4 --gacc 4 --bs 40

# bash scc_finetune_babylm.sh --lr 2e-4 --gacc 3 --bs 40 --ep 5
# bash scc_finetune_babylm.sh --lr 2e-4 --gacc 2 --bs 40 --ep 5
# bash scc_finetune_babylm.sh --lr 2e-4 --gacc 4 --bs 40 --ep 5

# bash scc_finetune_babylm.sh --lr 2e-4 --gacc 3 --bs 40 --ep 1
# bash scc_finetune_babylm.sh --lr 2e-4 --gacc 3 --bs 40 --ep 10

# ============ phase 1 ============
# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 80 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 80 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 80 --ratio 75 --ep 8

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 40 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 40 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 40 --ratio 75 --ep 8

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 2 --bs 80 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 2 --bs 80 --ratio 75 --ep 8
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 2 --bs 80 --ratio 75 --ep 8

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 80 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 80 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 80 --ratio 50 --ep 12

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 40 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 40 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 40 --ratio 50 --ep 12

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 2 --bs 80 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 2 --bs 80 --ratio 50 --ep 12
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 2 --bs 80 --ratio 50 --ep 12

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 80 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 80 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 80 --ratio 25 --ep 24

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 1 --bs 40 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 1 --bs 40 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 1 --bs 40 --ratio 25 --ep 24

# bash scc_phase1_babylm_distill.sh --lr 1e-3 --gacc 2 --bs 80 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-3 --gacc 2 --bs 80 --ratio 25 --ep 24
# bash scc_phase1_babylm_distill.sh --lr 3e-4 --gacc 2 --bs 80 --ratio 25 --ep 24

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 64
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 64
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 64

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 32
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 32
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 32

# bash scc_phase1_babylm.sh --lr 1e-3 --gacc 1 --bs 128
# bash scc_phase1_babylm.sh --lr 3e-3 --gacc 1 --bs 128
# bash scc_phase1_babylm.sh --lr 3e-4 --gacc 1 --bs 128

# ============ phase 2 ============
# learning_rates=("2e-3" "2e-4" "2e-5")
# batch_sizes=(32 64 128)
# epochs=(5 10)
# for ep in "${epochs[@]}"; do
#     for bs in "${batch_sizes[@]}"; do
#         for lr in "${learning_rates[@]}"; do
#             bash scc_phase2_babylm.sh --lr $lr --gacc 1 --bs $bs --ep $ep
#         done
#     done
# done

# bash scc_phase2_babylm.sh --lr 2e-4 --gacc 1 --bs 32 --ep 10
# bash scc_phase2_babylm.sh --lr 2e-4 --gacc 1 --bs 32
# bash scc_phase2_babylm.sh --lr 2e-4 --gacc 1 --bs 32

# bash scc_phase2_babylm.sh --lr 2e-5 --gacc 1 --bs 16
# bash scc_phase2_babylm.sh --lr 2e-5 --gacc 1 --bs 8

# qsub scc_phase2_babylm_distill.sh --lr 2e-4 --gacc 1 --bs 32
# qsub scc_phase2_babylm_distill.sh --lr 6e-5 --gacc 1 --bs 32
# qsub scc_phase2_babylm_distill.sh --lr 6e-4 --gacc 1 --bs 32

# qsub scc_phase2_babylm_distill.sh --lr 2e-4 --gacc 1 --bs 64
# qsub scc_phase2_babylm_distill.sh --lr 2e-4 --gacc 1 --bs 16
# qsub scc_phase2_babylm_distill.sh --lr 2e-4 --gacc 1 --bs 32 --ep 10

# ================== phase 2 distillation ratio =======================
# learning_rates=("2e-3" "2e-4" "2e-5")
# batch_sizes=(32 64 128)
# for bs in "${batch_sizes[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         bash scc_phase2_babylm_distill.sh \
#             --lr $lr \
#             --gacc 1 \
#             --bs $bs \
#             --ratio 75 \
#             --connector /projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/llava_babygpt_dino_phase1_random_mixed_data_75_bs160_lr3e-3_ep8/mm_projector.bin \
#             --ep 8
#     done
# done

# learning_rates=("2e-3" "2e-4" "2e-5")
# batch_sizes=(32 64 128)
# for bs in "${batch_sizes[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         bash scc_phase2_babylm_distill.sh \
#             --lr $lr \
#             --gacc 1 \
#             --bs $bs \
#             --ratio 50 \
#             --connector /projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/llava_babygpt_dino_phase1_random_mixed_data_50_bs160_lr1e-3_ep12/mm_projector.bin \
#             --ep 12
#     done
# done

# learning_rates=("2e-3" "2e-4" "2e-5")
# batch_sizes=(32 64 128)
# for bs in "${batch_sizes[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         bash scc_phase2_babylm_distill.sh \
#             --lr $lr \
#             --gacc 1 \
#             --bs $bs \
#             --ratio 25 \
#             --connector /projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/llava_babygpt_dino_phase1_random_mixed_data_25_bs640_lr3e-4_ep24/mm_projector.bin \
#             --ep 24
#     done
# done

# ====================== phase 1 & 2 tinyllama ===========================
# bash scc_phase1_babyllava_tinyllama.sh --lr 1e-3 --gacc 1 --bs 64
# bash scc_phase1_babyllava_tinyllama.sh --lr 3e-3 --gacc 1 --bs 64
# bash scc_phase1_babyllava_tinyllama.sh --lr 3e-4 --gacc 1 --bs 64
# echo "phase 1 done!!!!!!!!"

learning_rates=("2e-4" "2e-5" "2e-6")
batch_sizes=(32 64 128)
for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        bash scc_phase2_babylm.sh \
            --lr $lr \
            --gacc 1 \
            --bs $bs \
            --ep 5 \
            # --connector /projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/llava_vit_tinyllama_phase1_SAYCam_gt0.2_bs256_lr3e-4/mm_projector.bin \
    done
done
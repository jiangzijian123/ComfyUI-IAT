export CUDA_VISIBLE_DEVICES=6

# python3 -m musiq.run_predict_multi_image \
#   --ckpt_path="/gpfs/essfs/iat/Tsinghua/liwy/musiq/musiq_ckpts/musiq_spaq_ckpt.npz"

python3 -m musiq.run_predict_image \
  --ckpt_path="/gpfs/essfs/iat/Tsinghua/liwy/musiq/musiq_ckpts/musiq_spaq_ckpt.npz" \
  --image_path="/gpfs/essfs/iat/Tsinghua/liwy/musiq/1600x900.png"

# python3 -m musiq.run_predict_image \
#   --ckpt_path="/gpfs/essfs/iat/Tsinghua/liwy/musiq/musiq_ckpts/musiq_spaq_ckpt.npz" \
#   --image_path="/gpfs/essfs/iat/Tsinghua/liwy/musiq/512x256.png"
CUDA_VISIBLE_DECIVES=8
LOG_DIR="outputs/neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_demo_cvpr/neus-facto-wbg-reg_sep-plane-nerf/2024-03-15_165712"

MC_RES=512
MESH_FN="extracted_mesh_res-${MC_RES}.ply"
MESH_PATH="${LOG_DIR}/${MESH_FN}"

ns-extract-mesh \
	--load-config $LOG_DIR/config.yml \
    --load-dir $LOG_DIR/sdfstudio_models \
	--output-path $MESH_PATH \
    --chunk_size 25000 --store_float16 True \
    --resolution $MC_RES \
    --use_train_scene_box True \
    --seg_aware_sdf False \
    --remove_internal_geometry None \
    --remove_non_maximum_connected_components True \
    --close_holes False --simplify_mesh_final False --extract_texture True \


LOG_DIR="outputs/neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_demo_cvpr/neus-facto-angelo/2024-03-20_144633" 
MC_RES=512 
MESH_FN="extracted_mesh_res-${MC_RES}.ply" 
MESH_PATH="${LOG_DIR}/${MESH_FN}"
CUDA_VISIBLE_DEVICES=0 python scripts/texture.py --load-config $LOG_DIR/config.yml --input-mesh-filename $MESH_PATH --output-dir ./textures --target_num_faces 50000
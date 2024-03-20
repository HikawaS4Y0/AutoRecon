
DATA_ROOT=data
INST_REL_DIR=custom_data_example/toy
FORCE_RERUN=True
CUDA_VISIBLE_DEVICES=8

# Step 1 Coarse decomposition
python third_party/AutoDecomp/auto_decomp/cli/inference_transformer.py --config-name=cvpr \
    data_root=$DATA_ROOT \
    inst_rel_dir=$INST_REL_DIR \
    sparse_recon.n_images=40 \
    sparse_recon.force_rerun=$FORCE_RERUN \
    sparse_recon.n_feature_workers=1 sparse_recon.n_recon_workers=1 \
    triangulation.force_rerun=$FORCE_RERUN \
    triangulation.n_feature_workers=1 triangulation.n_recon_workers=1 \
    dino_feature.force_extract=$FORCE_RERUN dino_feature.n_workers=1 

# Step 2 train nerf model
# TODO: parse anno_dirname & object_filename from cache
CUDA_VISIBLE_DEVICES=8 ns-train neus-facto-angelo \
    --experiment-name neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_demo_cvpr \
    --vis tensorboard \
    --trainer.steps_per_eval_image 2500 \
    --trainer.steps_per_eval_batch 2500 \
    --trainer.max_num_iterations 60001 \
    --trainer.steps_per_save 10000 \
    --pipeline.datamanager.camera_res_scale_factor 0.25 \
    autorecon-data --data $DATA_ROOT/$INST_REL_DIR \
    --anno_dirname 'triangulate_loftr-720000_sequential_np-10/auto-deocomp_sfm-transformer_cvpr' \
    --camera_filename cameras_cameras_norm-obj-side-2.0.npz \
    --object_filename objects_cameras_norm-obj-side-2.0.npz \
    --parse_images_from_camera_dict True \
    --image_dirname "images" --image_extension ".jpg" \
    --include_image_features False --include_coarse_features False \
    --use_accurate_scene_box True \
    --collider_type 'box' \
    --near_far 0.1 100.0 \
    --compute_fg_bbox_mask True \
    --force_recompute_fg_bbox_mask False \
    --decomposition_mode regularization \
    --downsample_ptcd -1

# Extract mesh with MC
LOG_DIR="outputs/neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_demo_cvpr/neus-facto-angelo/2024-03-20_144633"
MC_RES=512
MESH_FN="extracted_mesh_res-${MC_RES}.ply"
MESH_PATH="${LOG_DIR}/${MESH_FN}"

# step 3 extract mesh
ns-extract-mesh \
	--load-config $LOG_DIR/config.yml \
	--output-path $MESH_PATH \
    --resolution $MC_RES \
    --simplify_mesh False 
 
# step 4 generate textural mesh
CUDA_VISIBLE_DEVICES=0 python scripts/texture.py --load-config $LOG_DIR/config.yml --input-mesh-filename $MESH_PATH --output-dir ./textures/new --target_num_faces 50000
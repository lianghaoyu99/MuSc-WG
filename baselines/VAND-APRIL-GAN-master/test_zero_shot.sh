### test on the VisA dataset
# python test.py --mode zero_shot --dataset visa \
# --data_path ./data/visa --save_path ./results/visa/zero_shot \
# --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/mvtec_pretrained.pth \
# --model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518

### test on the MVTec AD dataset
cd G:/MuSc-main/baselines/VAND-APRIL-GAN-master
python data/mvtec.py
python test.py --mode zero_shot --dataset mvtec \
--data_path ../../data/mvtec_anomaly_detection --save_path ../../output/VAND-APRIL-GAN-master/mvtec/zero_shot \
--config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/visa_pretrained.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518

### test on the MicroLED dataset
cd G:/MuSc-main/baselines/VAND-APRIL-GAN-master
python data/microled.py
python test.py --mode zero_shot --dataset microled \
--data_path ../../data/microled_AD --save_path ../../output/VAND-APRIL-GAN-master/microled/zero_shot \
--config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/mvtec_pretrained.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518

### test on the MiniLED dataset
cd G:/MuSc-main/baselines/VAND-APRIL-GAN-master
python data/miniled.py
python test.py --mode zero_shot --dataset miniled \
--data_path ../../data/miniled_AD --save_path ../../output/VAND-APRIL-GAN-master/miniled/zero_shot \
--config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/mvtec_pretrained.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518

for BETA in 0.005
do
for db_name in cifar100 imagenet-r
do
for forward_times in 120
do
for gamma in 0.1
do
for model in coop_variational_sr maple_variational_sr clclip_var_sr
do
python3 main_incremental_submit.py --lasp --beta 15 --db_name $db_name --use-vga --expandable-adapter --finetuning --finetune-epochs 2 --num-run 10 --compute-ece --compute-bwt --train_batch 20 --root ./mammoth_datasets/ --multi-gpu --gpus 0 --default-gpu 0 --model $model --exemplar-selector random --arch ViT-B/16 --epochs 5 --forward-times $forward_times  --method er --variational --get-adapter-distances  --sr-beta $BETA --gamma $gamma
done
done
done
done
done

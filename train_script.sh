# lr=1e-3
# batch_size=120
# nipe_values=(4250 8500)
# epoch_values=(20 60 100)

# for nipe in "${nipe_values[@]}";
# do
#   for epoch in "${epoch_values[@]}";
#   do
#     echo "Baseline-lr:${lr}-NIPE:${nipe}-Epoch:${epoch}-Batch:${batch_size}" >> result.log
#     torchrun --nproc_per_node=4 main.py --lr=${lr} --batch-size=${batch_size} --nipe=${nipe} --num-epoch=${epoch} --writer-name="lr:${lr}-NIPE:${nipe}-Epoch:${epoch}-Batch:${batch_size}"
#   done
# done

torchrun --nproc_per_node=4 main.py --lr=1e-3 --batch-size=60 --nipe=4250 --num-epoch=60 --only-eval --quant-eval
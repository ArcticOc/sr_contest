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
# 2 9 11 -- 3 12
# torchrun --nproc_per_node=4 main.py --lr=1e-3 --batch-size=20 --nipe=4250 --num-epoch=100 --model-type=model4
# torchrun --nproc_per_node=4 main.py --lr=1e-4 --batch-size=100 --nipe=4250 --num-epoch=60 --model-type=model11
# torchrun --nproc_per_node=4 main.py --only-eval --model-type=model12


model=(1 2 3 4 5 9 11)
for m in ${model[@]};
do
    echo "Model${m}_result" >> result.log
    torchrun --nproc_per_node=4 main.py --lr=1e-3 --batch-size=15 --nipe=8500 --num-epoch=120 --model-type=model${m}
done

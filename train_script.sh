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
# torchrun --nproc_per_node=4 main.py --only-eval --model-type=model5


# model=(4)
# epoch=(100 120)
# nipe=(4250 8500)
# lr=('1e-3' '1e-4')
# for e in ${epoch[@]};
# do
#     for n in ${nipe[@]};
#     do
#         for l in ${lr[@]};
#         do
#             for m in ${model[@]};
#             do
#                 echo "Model${m}_lr:${l}-NIPE:${n}-Epoch:${e}" >> result.log
#                 torchrun --nproc_per_node=4 main.py --lr=${l} --batch-size=20 --nipe=${n} --num-epoch=${e} --model-type=model${m}
#             done
#         done
#     done
# done
model=(2)
loss=('MSELoss' 'L1Loss' 'VGGPerceptualLoss' 'AdversarialLoss' 'CharbonnierLoss' 'EdgeLoss' 'CombinedLoss')
for m in ${model[@]};
do
    for l in ${loss[@]};
    do
        echo "Model${m}_result with loss${l}" >> result.log
        torchrun --nproc_per_node=4 main.py --lr=1e-3 --batch-size=15 --nipe=8500 --num-epoch=120 --model-type=model${m} --loss=${l}
    done
done

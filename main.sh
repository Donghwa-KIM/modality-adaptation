

EPOCHS=5
BATCH_SIZE=32

for N_SNAPSHOT in 1 5
do
    for ATTENTION_DROPOUT in 0.0 0.3 0.5 
    do
        for lr in 0.1 0.001
        do
        echo "
            --ATTENTION_DROPOUT=${ATTENTION_DROPOUT}
            --learning_rate=${BATCH_SIZE}
            --n_snapshot=${N_SNAPSHOT}
            "


        python main.py --lr=${lr} \
            --attn_dropout_a=${ATTENTION_DROPOUT} \
            --attn_dropout_t=${ATTENTION_DROPOUT} \
            --num_epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE}\
            --n_snapshot=${N_SNAPSHOT} \
            --adaptation \
            --late_fusion 
        python main.py --lr=${lr} \
            --attn_dropout_a=${ATTENTION_DROPOUT} \
            --attn_dropout_t=${ATTENTION_DROPOUT} \
            --num_epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE}\
            --n_snapshot=${N_SNAPSHOT} \
            --late_fusion 

        python main.py --lr=${lr} \
            --attn_dropout_a=${ATTENTION_DROPOUT} \
            --attn_dropout_t=${ATTENTION_DROPOUT} \
            --num_epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE}\
            --n_snapshot=${N_SNAPSHOT} \
            --adaptation 


        python main.py --lr=${lr} \
            --attn_dropout_a=${ATTENTION_DROPOUT} \
            --attn_dropout_t=${ATTENTION_DROPOUT} \
            --num_epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE}\
            --n_snapshot=${N_SNAPSHOT}
        done
    done
done
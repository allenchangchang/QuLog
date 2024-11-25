nohup python main_run.py --folder=bgl/ --log_file=BGL.log --dataset_name=bgl --model_name=qlstm --device=cuda --window_type=sliding --sample=sliding_window --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=4096 --num_candidates=150 --history_size=100 --lr=0.001 --n_qubits=4 --n_qlayers=1 --accumulation_step=5 --session_level=hour --window_size=60 --step_size=60 --output_dir=experimental_results/bgl/qdeeplog_4bits_1layers/ --is_process > ./log/bgl_qdeeplog_4bit_batchsize4096_1layers.out 2>&1 &
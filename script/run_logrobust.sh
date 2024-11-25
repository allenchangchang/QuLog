python main_run.py --folder=bgl/ --log_file=BGL.log --dataset_name=bgl --model_name=logrobust --device=cuda:0 --window_type=sliding --sample=sliding_window --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=1024 --num_candidates=150 --history_size=10 --lr=0.001 --accumulation_step=5 --session_level=hour --window_size=60 --step_size=60 --output_dir=experimental_results/logrobust/ --input_size=300 --semantics --is_process
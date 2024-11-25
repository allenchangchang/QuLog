# Quantum Machine Learning in Log-based Anomaly Detection: Challenges and Opportunities
<!-- ### **Under extension. Please refer the [dev](https://github.com/LogIntelligence/LogADEmpirical/tree/dev) branch.** -->
**Abstract**: 
Log-based anomaly detection (LogAD) is the main component of Artificial Intelligence for IT Operations (AIOps), which can detect anomalous that occur during the system on-the-fly. Existing methods commonly extract log sequence features using classical machine learning techniques to identify whether a new sequence is an anomaly or not. However, these classical approaches often require trade-offs between efficiency and accuracy. The advent of quantum machine learning (QML) offers a promising alternative. By transforming parts of classical machine learning computations into parameterized quantum circuits (PQCs), QML can significantly reduce the number of trainable parameters while maintaining accuracy comparable to classical counterparts. 
In this work, we introduce a unified framework, QuLog, for evaluating QML models in the context of LogAD. This framework incorporates diverse log data, integrated QML models, and comprehensive evaluation metrics. State-of-the-art methods such as DeepLog, LogAnomaly, and LogRobust, along with their quantum-transformed counterparts, are included in our framework.
Beyond standard metrics like F1 score, precision, and recall, our evaluation extends to factors critical to QML performance, such as specificity, the number of circuits, circuit design, and quantum state encoding. Using QuLog, we conduct extensive experiments to assess the performance of these models and their quantum counterparts, uncovering valuable insights and paving the way for future research in QML model selection and design for LogAD.
### Studied Models
| Model | Paper |
| :--- | :--- |
| DeepLog | [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3134015) |
| LogAnomaly | [LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs](https://www.ijcai.org/proceedings/2019/658) |
| LogRobust | [Robust log-based anomaly detection on unstable log data](https://dl.acm.org/doi/10.1145/3338906.3338931) |

### Requirements
  
The required packages are listed in requirements.txt. Install:

```
pip install -r requirements.txt
```

### Demo
- Example of QDeepLog on BGL with fixed window size of 1 hour:
```shell script
python main_run.py --folder=bgl/ --log_file=BGL.log --dataset_name=bgl --model_name=qlstm --device=cuda --window_type=sliding --sample=sliding_window --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=4096 --num_candidates=150 --history_size=100 --lr=0.001 --n_qubits=4 --n_qlayers=1 --accumulation_step=5 --session_level=hour --window_size=60 --step_size=60 --output_dir=experimental_results/bgl/qdeeplog_4bits_1layers/ --is_process
```
- For more explanation of parameters:
```shell script
python main_run.py --help
```
Running more model scripts can be found in `LogADEmpirical/script`
<!-- ## Citation
If you find the code and models useful for your research, please cite the following paper:
```
@inproceedings{le2022log,
  title={Log-based Anomaly Detection with Deep Learning: How Far Are We?},
  author={Le, Van-Hoang and Zhang, Hongyu},
  booktitle={2022 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)},
  year={2022}
}
``` -->

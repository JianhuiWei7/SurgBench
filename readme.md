This repo presents the official dataset release of SurgBench

# **File description:**

**SurgBench-E.json**: This json file details the clips of SurgBench-E. It includes the label index, path to the video clip, task type, duration, and other meta information.

**SurgBench-E_taxonomy.json**: This file details the taxonomy of our SurgBench-E, it includes 6 major categories, 10 sub categories, and 72 tasks.

**train.csv:** :This csv file only has two columns, the clip path and the label index. It facilates quick loading and usage

**test.csv:** :This csv file is the same format as described above.

**run_class_fine_tuning.py**: this file is the entry file for fine tuning and testing models on SurgBench-E

**run_mae_pre_training.py**: this file is the entry file for pretraining the foundation model on SurgBench-P

Other .py files serve as utils python file.

# **Segmentation**

**segment_clips_script**: This folder contains the file for how we split the original video into clips.

# Data

SurgBench-E: this folder conatins the video clips for pretraining
SurgBench-P: this folder contains the video clips for fine-tuning
We provide some examples in the folder above. For full video access, please refer to https://huggingface.co/datasets/JianhuiWei/SurgBench_NIPS25

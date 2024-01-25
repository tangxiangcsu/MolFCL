

 ## 🌈 Model Architecture
 ![Model_architecture](fig/Overview%20of%20MolFCL.png)

 ## 📚 Dataset Download
We have provided the pre-training and downstream datasets in the "data" folder.

## 📕 Code Path

#### Code Structures
There are four parts in the code.
- **chemprop**: It contains the main files for MolFCL network and training scripts for MolFCL.
- **data**: It contains the pre-training data and downstream dataset splits.
- **ckpt**: It saves checkpoint for pre-training.

## 🔬 Dependencies
- ```Python 3.7```
- ```PyTorch == 1.12.1+cu113```
- ```NumPy```
- All experiments are performed with one A100 GPU.

## 🚀 Train
The pre-training script:
```python
python pretrain.py
```

The finetuning script:
```python
python train.py
```

**Note**: 
- you can open the `train.py` file for parameter</a> modification.

## 🤗 email
If you have any questions please contact me xingtang@csu.edu.cn
## 🤝 Cite:
None

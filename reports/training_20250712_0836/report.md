# GPT Training Report

**Training Session:** `20250712_0836`
**Training Device:** `cpu`

## HYPERPARAMETERS

### **Model Architecture**

| Hyperparameter | Value |
|-----------|-------|
| seq_size | `8` tokens |
| batch_size | `32` |
| n_embd (dim) | `32` |
| num_heads | `4` |
| N_layers | `3` |
| dropout | `0` |

### **Training**

| Hyperparameter | Value |
|-----------|-------|
| training_steps | `100` |
| learning_rate | `0.001` |
| eval_interval | `10` steps |
| eval_iters | `10` |
| Train/Val Split | `90.0%` / `10.0%` |

## DATA PREPARATION

| Metric | Value |
|--------|-------|
| **Dataset** | `data/tinyshakespeare.txt` |
| **Vocabulary Size** | `65` tokens |
| **Training Tokens** | `1,003,854` tokens (90.0%)|
| **Validation Tokens** | `111,540` tokens (10.0%)|
| **Total Dataset Size** | `1,115,394` tokens |

## MODEL DETAILS

| Metric | Value |
|--------|-------|
| **Total Parameters** | `42,369` |
| **Trainable Parameters** | `42,369` |
| **Model Size** | ~`0.16` MB (float32) |
| **Optimizer** | AdamW with learning rate `0.001` |


## 🎯 Training Results

- **Final Training Loss:** `2.9256`
- **Final Validation Loss:** `2.9882`
- **Training duration:** `0:00:04.278747`

## Generated example:
```

F ry nN
Wipe adt se yuhole sIoirns ,etha:ss tod
th:loj E withinlao:dAtrl, h !dat dib Pnguuoure mb
j
St
ed.P Ls, a;i
DBsdeMnaitj tins maLaied Lk pheO 3ufi
jnaKet tZ: tiyiib taCas tainorir Ps:ones dulisOonlin
HMs pocmdo  fiiItMathYy caBLoUde lI 'kz oTGra
We sirs tsnd ciZnede:ct ,lE tcthRdokf aour-g: an s thal.N
nd ad roI ? otd,!metvtouer toKr i st tseigyooeat, l te .

TsDtple C Z:rot : raurefk Iwyt  p
Ae eoiH iciuDAeliceinne si ftweo yoweyhyt on ohd utyRWl tmlki mtci:ine Ae
Whaeq
Is geisethunoe Rm
```

## 📈 Training Progress

<img src="losses.png" alt="Training and Validation Loss" width="80%"/>


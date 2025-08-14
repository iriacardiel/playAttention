Welcome to the playAttention ! To see the full documentation, please visit the [Wiki](https://github.com/iriacardiel/playAttention/wiki).

# What is playAttention? 

This is a playground for understanding Attention and Transformers. For years, I felt that no matter how many videos or tutorials I watched, I never fully understood the architecture of Transformers at a low level. This repository is my way of getting hands-on experience by building my first Transformer-based language model. I hope it will be helpful to others who want to explore this fascinating field in the future.

# Resources

This is a list of videos, tutorials, and posts that have helped me throughout my learning journey. I recommend taking your time to go through them—they're worth a careful look.

**[Alfredo Canziani](https://atcold.github.io/)**
- [Class on Attention and Transformers [video]](https://www.youtube.com/watch?v=fEVyfT-gLqQ&t=828s)

**[Andrej Karpathy](https://karpathy.ai/)** 
- [Let's build GPT: from scratch, in code, spelled out [video]](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [nanGPT](https://github.com/karpathy/nanoGPT)

**Other**
- [Deep Learning.AI - How tranformer LLMs work [course]](https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/nfshb/introduction)
- [Jalammar - Illustrated Transformer [post]](https://jalammar.github.io/illustrated-transformer/)
- [Borealis - Tutorial #14: Transformers I: Introduction [post] ](https://rbcborealis.com/research-blogs/tutorial-14-transformers-i-introduction/)
- [Borealis - Tutorial #16: Transformers II: Extensions [post] * review after training with nanoGPT](https://rbcborealis.com/research-blogs/tutorial-16-transformers-ii-extensions/)
- [Borealis - Tutorial #17: Transformers III Training [post] * review after training with nanoGPT](https://rbcborealis.com/research-blogs/tutorial-17-transformers-iii-training/)

# Transformer Architecture (Decoder Only)

<p align="center">
  <img src="media/transformers.svg" width="750">
</p>

# The code repository

The code repository includes the implementation of both GPT and GPT-2 models, as well as the training scripts. The code is organized into several folders:

- **```models/```**: scritps ```model_GPT.py``` and ```model_GPT2.py``` contain the full architecture of the DIY-GPT models. It was built by following Karpathy's tutorials step by step, though you'll notice some differences in variable names, comments, refactoring, etc. I adapted it to what felt most intuitive for me—feel free to modify or build your own version as well.

-  **```train/```**: scritps ```train_GPT.py``` and ```train_GPT2.py``` load the configuration and the GPT models and launches the training loop. After training, an example of text generation will be executed, and a log files detailing the training process will be saved in the **```results/```** folder. For example, you can find the train / val loss plot that is generated during training:

<p align="center">
  <img src="/results/GPT_training_20250722_1758_cuda:0/losses.png"  width="750">
</p>

- **```Config.py```**: Defines the data model for the GPT models configuration, including hyperparameters and design choices related to the architecture. This configuration is necessary for loading and training the model. 


**Disclaimer**: This is an ongoing project—constantly evolving, growing, and being reviewed. As such, there may be mistakes, incomplete sections, or incorrect assumptions. Feedback and corrections are always welcome!

> **Environment** Setup

```
python -m venv venv
```

```
source venv/bin/activate && pip install -r requirements.txt
```

> **Training** GPT / GPT2 with **DDP** (supports single process):

```
torchrun --standalone --nproc_per_node=1 train/train_GPT2.py
```

```
torchrun --standalone --nproc_per_node=1 train/train_GPT.py
```

> **Training** GPT + GPT2 (sequentially):

```
make
```

![alt text](media/image.png)
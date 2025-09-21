This implementation of GPT2 follows the 124M model and is guided by Andrej Karpathy's recreation. It serves as a exploration of the outstanding architecture and features of GPT-2 rather than a strict replication.

My in-computer GPU is a NVIDIA GeForce MX330 which lacks many of the modern Pytorch runtime accelerations including torch.compile, Tensor Cores, and DistributedDataParallel. 

This means that I could not feasibly train this model on my machine. 

In addition, I have excluded pre-training on a sufficient dataset like FineWeb with 10B tokens. 


tokenizer.py is not used in the traingpt.py model, but serves as an example of a tokenizing process compared to tiktoken.

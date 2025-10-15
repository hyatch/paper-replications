# Paligemma 2B
This is a replication of the multimodal transformer released by Google that combines the Gemma 2B LLM and the SigLip Vision Transformer into a single model that is able to describe any given image. 

For example: ![](paligemma/citpsweikles.jpg)

The model when fed this image along with the prompt "The people are ", can output "The people are me and my friends sitting in front of a bauble." 
Some of the features in this model include
- Rotary Positional Embedding
- Multi Query Attention
- RMS Normalization

# Compress And Attend Transformers (CATs)

Worried about all the choices one needs to make for an efficient architecture!

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/8ea2a9dc-8dc9-4d11-ad79-67399d7ba68e" />

Enter CATs to the rescue!!

## Overview

- 🐈 CATs model _chunks of tokens_ given compressed representations of past chunks in the sequence
  
- 🗜️ Due to **_compression_**, FLOPs & KV-cache diminish by a factor of chunk size (upto **3x faster** and **7x memory efficient**)
  
- 📏 Choosing chunk size (i.e. how much to compress?) allows CATs to **interpolate** between compressed (fast) and dense (slow) transformer
  
- 😌 💆 No need to heuristically define efficient attention maps, No need to compose with other mixers to have competitive performance – **everything can be learned end-to-end** – offers a natural knob to control for the trade-off, where on one extreme, one recovers full attention (when `chunk_size` is 1)
  
- 🧑‍💻 No need for an efficient kernel for fast inference: We provide simple and efficient implementations for training and inference in pure PyTorch: utilizing basic attention implementations -- since CATs build on top of dense transformers!
  
- ✅ We provide single-file implementation for CATs
 


<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/9dfd3a04-a259-4a2d-a07e-e513f95b7710" />


Just choose chunk size (i.e. how much to compress?) and interpolate between compressed (fast) and dense (slow) transformer

<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/ae34c9e2-4267-4ea6-bb81-665cd62d8207" />

CATs are upto 3x faster and 7x memory efficient compared to a dense transformer!

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/6e7a1887-618f-4ef3-8809-a27b4d101ce3" />





---

This is official code for the paper: Autoregressive Language Modeling by Compressed Sequence Mixing

`cat_transformer.py` : contains hackable single-file implementation for the CAT transformer

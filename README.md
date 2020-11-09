# DeepHierProtein
Codes for *"Deep Hierarchical Embedding for Simultaneous Modeling of GPCR Proteins in a Unified Metric Space"*, submitted to *Bioinformatics*.

## Overall information
Model that learns a unified metric space for hierarchical protein families of G Protein Coupled Receptors is proposed in this work.


## Execution
Code for training the proposed model can be executed with following commands.
```
python main.py --seq_len 1000 \
			   --charset_size 20 \
			   --learning_rate 0.001 \
			   --max_epoch 200 \
			   --data_dir data \
			   --output_dir result \
			   --num_cuda 0
```


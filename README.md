Transformer architecture paper: https://arxiv.org/pdf/1706.03762

I tried to train the transformer on 80% of the input data in this repo for 10k iterations ( about 7 hours on my GPU ) and it loses generality.
4k iterations seems to be optimum. After that it seems like it's specializing on the training data at the cost of performing worse on data that it never sees.
Step 0,    Train loss: 4.2989, Val loss: 4.3010\n
Step 500,  Train loss: 1.6403, Val loss: 1.8056
Step 1000, Train loss: 1.4273, Val loss: 1.6292
Step 1500, Train loss: 1.3373, Val loss: 1.5638
Step 2000, Train loss: 1.2832, Val loss: 1.5326
Step 2500, Train loss: 1.2397, Val loss: 1.5277
Step 3000, Train loss: 1.1997, Val loss: 1.5200
Step 3500, Train loss: 1.1659, Val loss: 1.5197
_**Step 4000, Train loss: 1.1353, Val loss: 1.5156**_
Step 4500, Train loss: 1.0998, Val loss: 1.5278
Step 5000, Train loss: 1.0713, Val loss: 1.5412
Step 5500, Train loss: 1.0408, Val loss: 1.5453
Step 6000, Train loss: 1.0098, Val loss: 1.5625
Step 6500, Train loss: 0.9799, Val loss: 1.5792
Step 7000, Train loss: 0.9498, Val loss: 1.5979
Step 7500, Train loss: 0.9196, Val loss: 1.6212
Step 8000, Train loss: 0.8922, Val loss: 1.6365
Step 8500, Train loss: 0.8607, Val loss: 1.6663
Step 9000, Train loss: 0.8361, Val loss: 1.6850
Step 9500, Train loss: 0.8097, Val loss: 1.7012
Step 9999, Train loss: 0.7846, Val loss: 1.7282

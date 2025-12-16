# PCDVQ

Implementation of the paper on PCDVQ

## Results

#### QWEN3-0.6B

| Metric          | Original |   Quantized |
| --------------- | -------: | ----------: |
| bits_per_byte   |   0.9115 |      1.5815 |
| byte_perplexity |   1.8810 |      2.9928 |
| word_perplexity |  29.3252 | 351.4114 |


#### QWEN3-8B

|    Metric     | Original | Quantized |
|---------------|---------:|---------: |
|bits_per_byte  |    0.7015|     1.6540|
|byte_perplexity|    1.6262|     3.1471|
|word_perplexity|   13.4642|   459.8194|


 #### Simple model (Linear on MNIST)

| Codebook bits| Accuracy e8 | Accuracy e8P |
|-------------:|------------:| -----------: |
| W/O          |    0.863783 |    0.863783  |
| 8_2          |    0.826283 |    0.779383  |
| 10_2         |    0.838266 |    0.796283  |
| 12_2         |    0.849016 |    0.817733  |
| 14_2         |    0.85315  |    0.84685   |
| 16_2         |    0.855466 |    0.85496   |

# LLaRD

This is pytorch implemention for our LLaRD:

## Environment Requirement

The required packages are as follows:

- python == 3.9.12
- pytorch == 2.0.0
- numpy == 1.26.4
- pandas == 2.2.2
- numba == 0.60.0
- json5 == 0.9.25
- faiss-gpu == 1.7.1

## Datasets

You can download original dataset from the following links:

Steam: https://github.com/kang205/SASRec

Yelp: https://www.yelp.com/dataset

Amazon-Book: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html


We also provide the processed dataset [amazon-book][amazon-book] and [yelp][yelp] in the `data` folder.

This follow the Text-attributed Recommendation Dataset from [[RLMRec](https://arxiv.org/abs/2310.15950)], which includes description text, profile information and text embedding.

The files `usr_emb_np.pkl` and `itm_emb_np.pkl` can also be obtained from [RLMRec]

Additionally, we construct denoising knowledge, facilitated by the Qwen model.

To aid in understanding our denoising knowledge mining process and more details in the chain-of-thought reasoning, we provide a system prompt example in the `data` folder.

## Parameters

Key parameters are mentioned in our paper. More related parameters are all listed in the `parse_args` function.

## Quick Start

The command to implemented LLaRD is as follows:

```bash
python run_LLaRD.py

```

[Amazon-Book]: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
[Steam]: https://github.com/kang205/SASRec
[Yelp]: https://www.yelp.com/dataset
[Amazon-Book]: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
[Yelp]: https://www.yelp.com/dataset
[Amazon-Book]: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

# CAFE

This repository contains the code of model CAFE.

Our SIGIR 2022 paper [Coarse-to-Fine Sparse Sequential Recommendation](https://arxiv.org/pdf/2204.01839.pdf).

# Overview

Sequential recommendation aims to model dynamic user behavior from historical interactions. Self-attentive methods have proven effective at capturing short-term dynamics and long-term preferences. Despite their success, these approaches still struggle to model sparse data, on which they struggle to learn high-quality item representations. We propose to model user dynamics from shopping intents and interacted items simultaneously. The learned intents are coarse-grained and work as prior knowledge for item recommendation. To this end, we present a coarse-to-fine self-attention framework, namely CaFe, which explicitly learns coarse-grained and fine-grained sequential dynamics. Specifically, CaFe first learns
intents from coarse-grained sequences which are dense and hence provide high-quality user intent representations. Then, CaFe fuses intent representations into item encoder outputs to obtain improved item representations. Finally, we infer recommended items based on representations of items and corresponding intents.

# Dataset
You can download Tmall dataset used in our experiment from [here](https://drive.google.com/file/d/1kdEyYDVAk8gvidwF34kUi3IsiX4tJAbR/view?usp=sharing).

Download the `.zip` file and unzip to the `data` folder.

# Training Scripts

We provide example training scripts to trian CAFE on Tmall dataset:

```bash
bash script/train_cafe.sh $gpu_id$ tmall
```
Our code will output the evalutation results on the test set to the console.

All training arguments can be found at `src/utils/options.py`

# Contact

If you have any questions related to the code or the paper, feel free to email Jiacheng (`j9li@eng.ucsd.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

# Citation

Please cite our paper if you use CAFE in your work:

```bibtex
@article{Li2022CoarsetoFineSS,
  title={Coarse-to-Fine Sparse Sequential Recommendation},
  author={Jiacheng Li and Tong Zhao and Jin Li and Jim Chan and Christos Faloutsos and George Karypis and Soo-Min Pantel and Julian McAuley},
  journal={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}
```

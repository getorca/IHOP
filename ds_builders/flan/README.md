# IHOP FLAN(esque) Builder

This is a series of scripts for recreating the FLAN datasets that are "FLANesque". They use the same root datasets, but some of the prompt templates are slightly different, but this should have no noticable effect on the final datasets, and some of the templates could be better.

## Usage

So far only FLAN CoT has been implemented.

Download the dataset @ <https://huggingface.co/datasets/winddude/IHOP_flan_CoT>

### Recreate from scratch

1) Clone this repo. Will need git LFS.
2) Install Polars `pip install polars` or install the requirements.txt located in the parent folder.
3) Recreate the dataset with the code in `ihop_flan_cot.ipynb` in this directory. Just run all the cells.

## Why recreate FLAN?

Honestly the FLAN repo is a cluster fuck. Including:

- it's built to create training data for the FALN tensorflow models, not creating datasets, so it does a lot more than it needs to, like toeknization.
  - due to the layers of abstraction inside of FLAN the toeknization is really hard to remove
- Abstraction to the max. FLAN is a messy abstraction on top of seqIO, which is an abstraction on Tensorflow Datasets which I believe is an abstraction on arrow. These are all made to work for tensforflow with the goal of tokenization.
- Arrow is honestly slow in python compared to polars. This is probably applified by seqqIO and tasks(python functions), constantly needing to switch between python and other languages.
- The templates file has lots of duplication and can be much neater.
- Getting rid of seqIO ellimates a lot of the TASK boilplate abstration, and any need to tokenize.

So at the end of the day the FLAN repo is expensive and impractical to work with.

## Key difference in the Dataset

- the `target` from the original dataset is included. This is important for verify synthetic data created by models.
- CoT responses have double line breaks between every "thought"/"step". This can easily be commented out in the notebook if you wish. But it was done to make it easier to eval with something like PRM outlined as outlined in "Let’s Verify Step by Step"
- It's only zeroshot.
- not labeled as OPT NOOPT. Looking at FLAN, only `synth_cot_.*` templates have options. These datasets are not available. The root CoT datasets are already formated with the options correctly. Although theoritcally we could strip options where they exists. But the mix is already nice. It also appears FLAN incorrectly labels some NOOPT as OPT.

## Citations

```
@article{longpre2023flan,
  title={The Flan Collection: Designing Data and Methods for Effective Instruction Tuning},
  author={Longpre, Shayne and Hou, Le and Vu, Tu and Webson, Albert and Chung, Hyung Won and Tay, Yi and Zhou, Denny and Le, Quoc V and Zoph, Barret and Wei, Jason and others},
  journal={arXiv preprint arXiv:2301.13688},
  year={2023}
}


@article{lightman2023lets,
      title={Let's Verify Step by Step}, 
      author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
      journal={arXiv preprint arXiv:2305.20050},
      year={2023}
}

@article{konstantin2023an,
      title={An automatically discovered chain-of-thought prompt generalizes to novel models and datasets}, 
      author={Konstantin Hebenstreit, Robert Praas, Louis P Kiesewetter, Matthias Samwald},
      journal={arXiv preprint arXiv:2305.02897},
      year={2023}
}

```



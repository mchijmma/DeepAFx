# [DeepAFX: Deep Audio Effects](https://mchijmma.github.io/DeepAFx/)

Audio signal processing effects (FX) are used to manipulate sound characteristics across a variety of media. Many FX, however, can be difficult or tedious to use, particularly for novice users. In our work, we aim to simplify how audio FX are used by training a machine to use FX directly and perform automatic audio production tasks. By using familiar and existing tools for processing and suggesting control parameters, we can create a unique paradigm that blends the power of AI with human creative control to empower creators. For a quick demonstration, please see our demo video:

[![Demo Video](https://github.com/adobe-research/DeepAFx/blob/main/images/video.png?raw=true)](https://youtu.be/6ujkPwcQKo4)

To combine deep learning and audio plugins together, we have developed a new method to incorporate third-party, audio signal processing effects (FX) plugins as layers within deep neural networks. We then use a deep encoder to analyze sounds and learn to control audio FX that themselves performs signal manipulation. To train our network with non-differentiable FX layers, we compute FX layer gradients via a fast, parallel stochastic approximation scheme within a standard automatic differentiation graph, enabling efficient end-to-end backpropagation for deep learning training. For technical details of the work, please see:


"[Differentiable Signal Processing with Black-Box Audio Effects.](https://mchijmma.github.io/DeepAFx/)"
[Marco A. Martínez Ramírez](https://m-marco.com/about/), [Oliver Wang](http://www.oliverwang.info/), [Paris Smaragdis](https://paris.cs.illinois.edu/), and [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/). 
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2021.

>@inproceedings{martinez2021deepafx,<br />
>   title={Differentiable Signal Processing with Black-Box Audio Effects},<br />
>   author={Mart\'{i}nez Ram\'{i}rez, Marco A. and Wang, Oliver and Smaragdis, Paris and Bryan, Nicholas J.},<br />
>   booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},<br />
>   month={June},<br />
>   year={2021},<br />
>   publisher={IEEE}<br />
>}<br />

# ICASSP Presentation Materials

The video is avaliable [here](https://www.youtube.com/watch?v=8pBDRxNOLU4) and the poster is available [here](https://marquetem.files.wordpress.com/2021/05/icassp_poster_submission.pdf).

# Code

All code and models are available [here](https://github.com/adobe-research/DeepAFx).

# Listening Test Examples


All listening test examples are available [here](./LISTENING_TEST.md).

&nbsp;
&nbsp;
&nbsp;




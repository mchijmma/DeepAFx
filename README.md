# DeepAFX: Deep Audio Effects 

Audio signal processing effects (FX) are used to manipulate sound characteristics across a variety of media. Many FX, however, can be difficult or tedious to use, particularly for novice users. In our work, we aim to simplify how audio FX are used by training a machine to use FX directly and perform automatic audio production tasks. For a quick demonstration, please see our demo video:

[![Demo Video](https://github.com/adobe-research/DeepAFx/blob/main/images/video.png?raw=true)](https://youtu.be/6ujkPwcQKo4)

<!--Our goal is to make the process of audio effects control easier and more powerful for audio content creators. To address this, we are investigating how to use deep neural networks/AI to control audio plugins (e.g. VST, AU, LV2 effects) or black-box audio effects, which themselves are used to perform audio processing. By using familiar and existing tools for processing and suggesting control parameters, we can create a unique paradigm that blends the power of AI with human creative control to empower creators.-->


To combine deep learning and audio plugins together, we have developed a new method to incorporate third-party, audio signal processing effects (FX) plugins as layers within deep neural networks. We then use a deep encoder to analyze sounds and learn to control audio FX that themselves performs signal manipulation. To train our network with non-differentiable FX layers, we compute FX layer gradients via a fast, parallel stochastic approximation scheme within a standard automatic differentiation graph, enabling efficient end-to-end backpropagation for deep learning training. For technical details of the work, please see:


"[Differentiable Signal Processing with Black-Box Audio Effects.](https://mchijmma.github.io/DeepAFx/)"
[Marco A. Martínez Ramírez](https://m-marco.com/about/), [Oliver Wang](http://www.oliverwang.info/), [Paris Smaragdis](https://paris.cs.illinois.edu/), and [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/). 
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2021.

# Code

[View the source code.](https://github.com/adobe-research/DeepAFx)


# Listening Test Examples

### Tube amplifier emulation
<div id="contentBox" style="margin:0px auto; width:385%">
<div id="column1" style="float:left; margin:0; width:15.75%;">
- Low-anchor <br />
<audio controls="controls">
    <source src="audio/distortion/full_model_61_101_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_99_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_84_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_69_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_44_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_35_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_31_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_6_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_5_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_2_x.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column2" style="float:left; margin:0;width:15.75%;">
- Mid-anchor <br />
<audio controls="controls">
    <source src="audio/distortion/full_model_61_101_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_99_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_84_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_69_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_44_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_35_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_31_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_6_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_5_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_2_x_mid.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column3" style="float:left; margin:0;width:15.75%">
- High-anchor <br />
<audio controls="controls">
    <source src="audio/distortion/full_model_61_101_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_99_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_84_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_69_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_44_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_35_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_31_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_6_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_5_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_2_y.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column4" style="float:left; margin:0;width:15.75%">
- DeepAFx <br />
<audio controls="controls">
    <source src="audio/distortion/full_model_61_101_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_99_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_84_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_69_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_44_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_35_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_31_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_6_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_5_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/full_model_61_2_z.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column5" style="float:left; margin:0;width:15.75%;">
- <a href="https://www.mdpi.com/2076-3417/10/2/638">CAFx</a><br />
<audio controls="controls">
    <source src="audio/distortion/CAFx_101.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_99.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_84.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_69.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_44.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_35.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_31.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_6.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_5.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/distortion/CAFx_2.mp3" type="audio/mp3" />
</audio>
</div>

</div>
&nbsp;

### Automatic non-speech sound removal
<div id="contentBox" style="margin:0px auto; width:385%">
<div id="column1" style="float:left; margin:0; width:15.75%;">
- Low-anchor <br />
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_0_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_1_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_2_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_3_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_4_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_5_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_6_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_7_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_8_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_9_x.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column2" style="float:left; margin:0;width:15.75%;">
- Mid-anchor <br />
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_0_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_1_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_2_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_3_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_4_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_5_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_6_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_7_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_8_x_mid.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_9_x_mid.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column3" style="float:left; margin:0;width:15.75%">
- High-anchor <br />
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_0_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_1_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_2_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_3_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_4_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_5_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_6_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_7_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_8_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_9_y.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column4" style="float:left; margin:0;width:15.75%">
- DeepAFx <br />
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_0_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_1_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_2_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_3_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_4_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_5_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_6_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_7_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_8_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/nonspeech/full_model_57_9_z.mp3" type="audio/mp3" />
</audio>
</div>

</div>
&nbsp;

### Automatic music mastering
<div id="contentBox" style="margin:0px auto; width:385%">
<div id="column1" style="float:left; margin:0; width:15.75%;">
- Low-anchor <br />
<audio controls="controls">
    <source src="audio/mastering/full_model_92_0_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_1_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_2_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_3_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_4_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_5_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_6_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_7_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_8_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_9_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_10_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_11_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_12_x.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_13_x.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column2" style="float:left; margin:0;width:15.75%;">
- Mid-anchor <br />
<audio controls="controls">
    <source src="audio/mastering/full_model_92_0_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_1_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_2_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_3_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_4_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_5_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_6_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_7_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_8_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_9_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_10_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_11_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_12_x_mid_peak.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_13_x_mid_peak.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column3" style="float:left; margin:0;width:15.75%">
- High-anchor <br />
<audio controls="controls">
    <source src="audio/mastering/full_model_92_0_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_1_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_2_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_3_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_4_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_5_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_6_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_7_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_8_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_9_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_10_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_11_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_12_y.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_13_y.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column4" style="float:left; margin:0;width:15.75%">
- DeepAFx <br />
<audio controls="controls">
    <source src="audio/mastering/full_model_92_0_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_1_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_2_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_3_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_4_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_5_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_6_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_7_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_8_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_9_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_10_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_11_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_12_z.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_13_z.mp3" type="audio/mp3" />
</audio>
</div>

<div id="column4" style="float:left; margin:0;width:15.75%">
- LANDR <br />
<audio controls="controls">
    <source src="audio/mastering/full_model_92_0_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_1_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_2_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_3_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_4_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_5_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_6_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_7_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_8_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_9_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_10_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_11_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_12_x_landr.mp3" type="audio/mp3" />
</audio>
<audio controls="controls">
    <source src="audio/mastering/full_model_92_13_x_landr.mp3" type="audio/mp3" />
</audio>
</div>

</div>
&nbsp;


Please use Mozilla Firefox or Google Chrome.
This project is maintained by <a href="https://m-marco.com">{{ "Marco Martínez" }}</a>.


<!-- &nbsp;
### Citation
>@aticle{martinez2020deep,<br />
>   title={Deep Learning for Black-Box Modeling of Audio Effects},<br />
>   author={Mart\'{i}nez Ram\'{i}rez, Marco A and Benetos, Emmanouil and Reiss, Joshua D},<br />
>   journal={Applied Sciences},<br />
>   volume={10},<br />
>   number={2},<br />
>   pages={638},<br />
>   month={January},<br />
>   year={2020},<br />
>   publisher={Multidisciplinary Digital Publishing Institute}<br />
>}<br /> -->

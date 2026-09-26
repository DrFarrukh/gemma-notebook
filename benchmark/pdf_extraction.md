# PDF extraction benchmark

Legacy: `pypdf.Page.extract_text()`. New: `pymupdf4llm.to_markdown(page_chunks=True)`. Structured output chunk counts use paragraph and heading boundaries, with a 4000-character hard cap.

OCR note: the core PyMuPDF4LLM package is used without its optional Layout/OCR modules. Scanned pages continue through the app's existing image-only PDF detection.

## Chunk target comparison

The new extractor is run once per PDF. Chunk counts below compare 1800/250, 2800/350, and 3500/350 targets; the pypdf baseline uses 1800/250.

## 6eec7c41-1fbf-4e74-95b0-2abb37ad0c31-E2CNN_An_Efficient_Concatenated_CNN_for_Classification_of_Surface_EMG_Extracted_From_Upper_Limb.pdf

| Extractor / target | Pages | Characters | Chunks | Headings | Wall time (s) |
|---|---:|---:|---:|---:|---:|
| pypdf 1800/250 | 8 | 36547 | 27 | 0 | 1.367 |
| PyMuPDF4LLM 1800/250 | 8 | 38017 | 43 | 23 | 2.428 |
| PyMuPDF4LLM 2800/350 | 8 | 38017 | 37 | 23 | 2.428 |
| PyMuPDF4LLM 3500/350 | 8 | 38017 | 32 | 23 | 2.428 |

### PyMuPDF4LLM sample

#### Source page 1

IEEE SENSORS JOURNAL, VOL. 23, NO. 8, 15 APRIL 2023 8989

## E2CNN: An Efficient Concatenated CNN for Classification of Surface EMG Extracted From Upper Limb

Muhammad Farrukh Qureshi, Zohaib Mushtaq, Muhammad Zia ur Rehman,

and Ernest Nlandu Kamavuako, _Member, IEEE_

_**Abstract**_ **—Surface electromyography is a bioelectrical indi-**
**cator** **that** **emerges** **during** **muscle** **contraction** **and** **has** **been**
**widely** **used** **in** **a** **variety** **of** **clinical** **applications.** **Several**
**prosthetic** **control** **applications** **can** **benefit** **from** **the** **analy-**
**sis** **based** **on** **the** **classification** **of** **surface** **electromyography**
**(sEMG)** **signals.** **However,** **for** **the** **real-time** **application** **of**
**upper** **limb** **prosthesis,** **the** **EMG-based** **systems** **need** **robust**
**performance** **and** **rapid** **response** **behavior.** **In** **this** **study,**
**we** **propose** **an** **efficient** **concatenated** **convolutional** **neural**
**network** **(E2CNN)** **for** **classification** **of** **sEMG** **extracted** **from**
**the upper limb. We have tested and validated the performance**
**of** **the** **proposed** **E2CNN** **on** **two** **datasets:** **a** **longitudinal**
**dataset** **comprising** **ten** **nondisabled** **(healthy)** **subjects** **and**
**six** **transradial** **amputee** **subjects** **and** **spanning** **the** **data** **col-**
**lected** **for** **a** **period** **of** **seven** **days;** **and** **the** **publicly** **available**
**NinaPro** **DB1** **dataset.** **The** **raw** **sEMG** **signals** **are** **converted**
**into** **Log-Mel** **spectrograms** **(LMSs).** **This** **model** **combines** **the** **input** **layers** **with** **the** **output** **of** **each** **convolutional** **block**
**using** **concatenation** **layers.** **The** **proposed** **E2CNN** **when** **applied** **to** **LMS-based** **images** **provides** **a** **good** **response** **time**
**with** **high-performance** **accuracy** **of** **98.31%** **±** **0.5%** **and** **97.97%** **±** **1.41%** **for** **both** **nondisabled** **and** **amputee** **subjects.**
**When applied to NinaPro DB1, the proposed E2CNN has attained a mean accuracy of 91.27%, an increase by 24.67% with**
**respect** **to** **the** **baseline** **CNN** **model.** **The** **results** **show** **that** **the** **achieved** **results** **are** **comparable** **to** **those** **obtained** **using**
**stacked sparse autoencoders (SSAEs) and other CNN models; however, E2CNN is associated with reduced training and**
**prediction times, making it a potential candidate for real-time classification of sEMG based on LM spectrogram images.**

_**Index Terms**_ **— Concatenation, convolutional neural network (CNN), Log-Mel spectrogram (LMS), NinaPro DB1, surface**
**electromyography (sEMG).**

## I. INTRODUCTION

LECTROMYOGRAPHY, also known as EMG, is a biological signal that is frequently used for the purpose
E

Manuscript received 13 February 2023; revised 2 March 2023;
accepted 5 March 2023. Date of publication 15 March 2023; date of
current version 14 April 2023. The associate editor coordinating the
review of this article and approving it for publication was Dr. Varun Bajaj.
_(Corresponding author: Muhammad Farrukh Qureshi.)_

This work involved human subjects or animals in its research.
Approval of all ethical and experimental procedures and protocols
was granted by the Local Ethical Committee of Riphah International
University under Approval No.: Riphah/RCRS/REC/000121/20012016,
and performed in line with the Declaration of Helsinki.

Muhammad Farrukh Qureshi and Zohaib Mushtaq are with the
Department of Electrical Engineering, Riphah International University, Islamabad 44000, Pakistan (e-mail: muhammad.farrukh@
riphah.edu.pk; zohaib.mushtaq@riphah.edu.pk).

Muhammad Zia ur Rehman is with the Department of Biomedical Engineering, Riphah International University, Islamabad 44000,
Pakistan (e-mail: ziaur.rehman@riphah.edu.pk).

Ernest Nlandu Kamavuako is with the Centre for Robotics Research,
Department of Engineering, King’s College London, WC2G 4BG
London, U.K. (e-mail: ernest.kamavuako@kcl.ac.uk).

Digital Object Identifier 10.1109/JSEN.2023.3255408

of recognizing human motor gestures. This is an essential
component of human–computer interaction systems. EMG
signals have been the subject of extensive research and have
been implemented as a control input for upper limb prostheses, assistive wheel chairs, assistive humanoid robots, and
meal assistive robots [1], [2], [3], [4]. The control method
used in the development of a prosthesis determines the
device’s functionality, ease of use, and acceptability. EMGbased control systems for upper limb prosthesis have been
widely studied; however, implementing them in real-time
with high multiclass accuracy is still a challenge. Several
studies have used offline machine learning [5], [6], [7], [8]
and deep learning techniques [9], [10], [11], [12], [13],

[14], [15], [16] to classify EMG signals for upper limb
prosthesis.

Rehman et al. [9] used a single-layered convolutional neural
network (CNN) on a multiday sEMG dataset that was captured
for consecutive 15 days. The accuracy that was attained
with CNN was 97.60% ± 1.99. In [14], and CNNs and
multilayer perceptron (MLP) were applied to categorize the

1558-1748 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:18 UTC from IEEE Xplore. Restrictions apply.

#### Source page 2

8990 IEEE SENSORS JOURNAL, VOL. 23, NO. 8, 15 APRIL 2023

nine predetermined gestures. The average accuracy of CNN’s
gesture recognition is 99.47%, while the mean accuracy of
MLP’s gesture recognition is 98.42%. In their study, Huang
and Chen [17] used the Ninapro database, which included the
sEMG data of 40 subjects with 50 gestures. The accuracy
of their suggested technique, which included a spectrogram,
a CNN, and a long short-term memory (LSTM), was 80.929%
for the basic hand movements. Using a method that is based
on the short-time Fourier transform (STFT) representation
of EMG data and CNNs, Sengur et al. [18] were able to
attain an accuracy of 96.69%. Hajian and Morin [19] have
suggested two-stream CNN (TS-CNN) to acquire significant
features from raw EMG data using multiple scales, as well
as to estimate the motion that is created during elbow flexion
and extension. The results that were obtained were 81% ±
0.06, 71% ± 0.06, and 80% ± 0.13 for the estimate of
the joint angle, and 78% ± 0.05, 79% ± 0.07, and 71% ±
0.13 for the estimation of the velocity, during isotonic contractions, isokinetic contractions, and dynamic contractions,
respectively. Pancholi et al. [20] have tested the performance
of deep-learning-based pattern recognition (DLPR) framework
on features extracted from publicly available dataset NinaPro
databases and achieved an accuracy of 92.18%, 91.56%, and
84.98% on DB1, DB2, and DB3, respectively. The same
work also experimented on STFT spectrogram images from
publicly available dataset NinaPro databases and achieved an
accuracy of 70.14%, 74.89%, and 65.87% on DB1, DB2,
and DB3, respectively. Tuncer and Alkan [21] have identified
400 EMG spectrograms using hybrid deep learning approaches
that were based on transfer learning models such as AlexNet,
GoogleNet, and ResNet18. The accuracy for hybrid classification using AlexNet, GoogleNet, and ResNet18 was 99.17%,
95.83%, and 93.33%, respectively.
In this article, we propose a rapid, responsive deep neural
network (DNN) that has been applied to sEMG spectrogram
images. This DNN is based on the CNN that takes Log-Mel
spectrogram (LMS) images (extracted from EMG signals) as
an input. The significant contributions of this study are as
follows.

1) A custom nonsequential concatenated CNN is proposed
for classification of the upper limb.
2) Utilization of LMS images for sEMG signals for upper
limb gesture classification.
3) Analysis of the proposed an efficient concatenated CNN (E2CNN) on a longitudinal sEMG
dataset comprising ten nondisabled and six amputee
subjects.
4) Validation of the proposed technique on the publicly
available NinaPro DB1 dataset.
5) Performance comparison of the proposed E2CNN on
previous studies implemented on both the datasets.
The rest of the article is organized as follows. We present
the methodology containing the description of datasets, preprocessing steps, and details of the proposed DNN applied for
classification in Section II, discuss the results in Section III,
and finally provide the conclusion and future work in
Section V.

## II. METHODOLOGY
### A. Experimental Dataset and Setup Description

We have used two datasets in this study: 1) the first dataset
based on the longitudinal dataset from previous work [5]
and 2) the second dataset is the publicly available NinaPro
Dataset 1 (DB1). Details of both the datasets are discussed in
Sections II-A.1 and II-A.2.

#### 1. First Dataset:
The first dataset we used is from
previous work [5]. That dataset consists of ten nondisabled (healthy) and six transradial amputee subjects.
All the subjects were male and their mean ages were
24.5 ± 0.22 and 34.8 ± 0.32 years for nondisabled and
amputee subjects, respectively. An ethical approval was
taken prior to data collection (Approval No.: Ref No.
Riphah/RCRS/REC/000121/20012016). The authors simultaneously recorded the surface and intramuscular EMG. Six
surface and six intramuscular electrodes were used. An 8-kHz
sample rate is used to obtain data from six channeled sEMG
signals. Eleven distinct gestures (including rest) were performed by all the subjects with four repetitions of each
movement. The data were collected for seven consecutive
days. The hand gestures performed by participants are: rest,
open hand, closed hand, pronation, supination, fine grasp, side
grip, flex hand, extend hand, thumb up, and pointer and are
illustrated in Fig. 1(a). For the purpose of this study, we have
only used the sEMG dataset.

#### 2. Second Dataset:
The first NinaPro database (DB1)
includes 27 intact subjects [22]. This dataset includes 52
distinct movements carried out by 27 subjects, with each
movement having been performed ten times. These movements are categorized into three distinct types of exercises:
movements of the fingers; grasping and functional movements;
and isometric, isotonic hand configurations and basic wrist
movements. The data were collected using ten sEMG electrodes. For cohesion with the first dataset, we have only used
nine hand movements from NinaPro DB1. The gestures used
are: thumb up, hand close, hand open, pointer, supination,
pronation, flexion, extension, and wrist extension with closed
hand. Fig. 1(b) shows the gestures used from NinaPro DB1
for this study.

### B. Preprocessing Technique

The preprocessing technique is applied to both the datasets.
The EMG-based pattern recognition techniques require smaller
intervals of signals to extract useful classification features.
However, when the processing window decreases, the performance drops significantly [23]; hence, optimum duration is
limited between 150 and 250 ms [24], [25]. We divide each
raw EMG signal into smaller intervals (windows) with a length
of 200 ms and an overlapping increment of 29 ms. For the
first dataset, each EMG signal results in 4400 × 6 windowed
signals with data in six channels for 11 hand gestures. This
was done for all seven days for each subject and resulted in
30 800 × 6 windowed signals for each subject. For the second
dataset, each EMG signal results in 1791 × 10 windowed
signals with data in ten channels for nine hand gestures. To get
better interpretation of 1-D, stochastically distributed EMG

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:18 UTC from IEEE Xplore. Restrictions apply.

#### Source page 3

QURESHI et al.: E2CNN FOR CLASSIFICATION OF sEMG EXTRACTED FROM UPPER LIMB 8991

Fig. 1. Gestures used in this study. (a) Hand gestures used from the first dataset: rest, opened hand, closed hand, pronation, supination, fine
grasp, side grip, flex hand, extend hand, thumb up, and pointer. (b) Hand gestures used from NinaPro DB1: thumb up, hand close, hand open,
pointer, supination, pronation, flexion, extension, and wrist extension with closed hand.

where _H_ ∈ N is the hop length, _w_ : [0 : _N_ - 1] ∈ R is the
Hann window _w_ = 0 _._ 5 − 0 _._ 5 cos _((_ 2 _π_ _n/N_ - 1 _))_, _N_ ∈ N is
the length of _w_, and _x_ ∈[0 : _(L_ - _N_ _/H_ _)_ ] and _y_ ∈[0 : _(N_ _/_ 2 _)_ ]
indicate the time and frequency indices, respectively.

The STFT spectrogram of _Sw_ as illustrated in Fig. 2(b) can
be achieved by the following:

_S_ STFT _(x, y)_ = | _Sw(x, y)_ | [2] _._ (2)

The relationship between the Mel spectrum and the frequency is _f_ mel = 2959 × log10 _(_ 1 + _( f_ _/_ 700 _))_ . The LMS can
be estimated using the following:

_S_ LM _(x, y)_ =

_fc(x_ +1 _)_



_f (y)_ = _fc(x_ −1 _)_

log10 _(M(x, y)_ - _S_ STFT _(x, y))_ (3)

where _M(x, y)_ are the Mel filter banks and can be computed
from the following:

_M(x, y)_ =

 _f (y)_ - _fc(x_ - 1 _)_

for _fc(x_ - 1 _)_ ≤ _f (y)_
_fc(x)_ - _fc(x_ - 1 _)_ _[,]_

_<_ _fc(x)_



_f (y)_ - _fc(x_ + 1 _)_
for _fc(x)_ ≤ _f (y)_
_fc(x)_ - _fc(x_ + 1 _)_ _[,]_

_<_ _fc(x_ + 1 _)_

 0 others.

Fig. 2. Conversion of signals from the first dataset into LM spectrogram
images. (a) Windowed signal from one channel. (b) STFT of the windowed signal. (c) LMS. (d) Combined image of all the six channels as
LM spectrogram image.

signal, the windowed sEMG signals were converted into LMSs
using librosa package in python.

### C. Signals to Log-Mel Spectrograms

Let _sw(n)_ be an windowed EMG signal, with the length _L_
and sampling frequency _fsw_ in hertz as shown in Fig. 2(a).
Then its STFT _Sw_ will be

(4)

where _f (y)_ is the linear frequency and _fc(x)_ = _x_ - _δ_ _f_ mel are
the center frequencies on Mel-scale. Fig. 2(c) illustrates an
image of LM spectrogram for the EMG windowed signal _sw_ .

Each windowed signal is individually converted into an LM
spectrogram, and this process is repeated for each channel
resulting in six LM spectrograms. These six LM spectrograms
are then combined vertically and converted into an image as
shown in Fig. 2(d). As a result, we get 30 800 EMG images as
LM spectrograms as our input dataset to CNN for each subject.
The process for the first dataset is illustrated in Fig. 2. Similar
to the first dataset, the same technique is applied to the second
dataset and we get 1791 EMG images as LM spectrograms for
each subject with ten LM spectrograms combined vertically in
each image. The process is depicted in Fig. 3.

_Sw(x, y)_ =

_N_ −1

_n_ −0

_sw(n_ + _x H_ _)_ - _w(n)_ - _e_ [−] _[ι]_ [2] _[π]_ _[y]_ _N_ _[n]_ (1)

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:18 UTC from IEEE Xplore. Restrictions apply.

#### Source page 4

8992 IEEE SENSORS JOURNAL, VOL. 23, NO. 8, 15 APRIL 2023

Fig. 4. Block diagram of the proposed strategy implemented in this
article.

Fig. 3. Conversion of NinaPro DB1 signals into LM spectrogram
images. (a) Windowed signal from one channel. (b) STFT of the
windowed signal. (c) Logarithmic-Mel (LM) spectrogram. (d) Combined
image of all the ten channels as LM spectrogram image.

### D. E2CNN Architecture

The experimental process in this study involves a deep
CNN with max-pooling functions [26]. Let us name our
model as E2CNN. The proposed E2CNN is implemented on
sEMG-based LMS images with the input size of 128 × 128.

We have designed the proposed E2CNN to reduce the
number of parameters while maintaining performance. This
is achieved using convolutional layers with large kernel sizes
and max-pool layers with large pooling sizes. Consequently,
the number of parameters is reduced, but numerous features
are lost. Concatenation layers are used to mitigate with feature
loss. Concatenation layers append the original input to the
output of subsequent convolutional and max-pool layers. This
again combines the features that might be overlooked during
heavy padding and strides.

The total number of epochs in the network in E2CNN is
100, and the batch size is 32. For optimization, the Adam
optimizer is used. Except for the final layer, which uses the
Softmax activation function, all the layers of the proposed
E2CNN use the rectified linear unit (ReLU) activation function. The block diagram of the proposed E2CNN is illustrated
in Fig. 4. The model is divided into three sections: Feature
Block A, Feature Block B, and the Classification Block. Using
the rescaling layer (RL), the input images are resized to _(_ 0 _,_ 1 _)_ .

The following shows more information about the proposed
E2CNN:

#### 1. Feature Block A:

1) Layer 1a: The image from RL is fed to the model’s first
layer (L1a) having 16 filters, each with a 7 × 7 receptive
field. It is followed by a 2 × 2 strided max-pooling function with batch normalization in between. The activation
function used in this layer is ReLU.
2) Layer 2a: The second layer (L2a) consists of double
filters (32) in comparison to L1a with a reduced receptive field of 5 × 5 with a max-pooling with strides of
2 × 2 performed using ReLU as the activation function.
Similar to L1a, batch normalization is used.
3) Layer 3a: The third layer (L3a) is made up of double
filters (64) with respect to L2a with a further reduced
receptive field of 3 × 3. A max-pooling function with
strides of 2 × 2 using ReLU as the activation function
is applied following a batch normalization layer.

#### 2. Feature Block B:

1) Layer 1b + Concatenation: The input image passed
through RL is fed to this layer too. This layer (L1b)
has the same dimension and parameters as L1a. However, the output from RL is passed through another
max-pooling layer and combined with the output of L1b
using a concatenation layer (Concat1).
2) Layer 2b + Concatenation: The output from Concat1
becomes the input to this layer (L2b). The layer has
the same dimensions as L2a. Similar to L1b, the output

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:18 UTC from IEEE Xplore. Restrictions apply.

## 980eb62c-1dea-4209-9d8f-f6ff9d0a28ad-Spectral_Image-Based_Multiday_Surface_Electromyography_Classification_of_Hand_Motions_Using_CNN_for_HumanComputer_Interaction.pdf

| Extractor / target | Pages | Characters | Chunks | Headings | Wall time (s) |
|---|---:|---:|---:|---:|---:|
| pypdf 1800/250 | 8 | 34359 | 27 | 0 | 1.493 |
| PyMuPDF4LLM 1800/250 | 8 | 33093 | 35 | 20 | 2.862 |
| PyMuPDF4LLM 2800/350 | 8 | 33093 | 32 | 20 | 2.862 |
| PyMuPDF4LLM 3500/350 | 8 | 33093 | 30 | 20 | 2.862 |

### PyMuPDF4LLM sample

#### Source page 1

20676 IEEE SENSORS JOURNAL, VOL. 22, NO. 21, 1 NOVEMBER 2022

## Spectral Image-Based Multiday Surface Electromyography Classification of Hand Motions Using CNN for Human–Computer Interaction

Muhammad Farrukh Qureshi, Zohaib Mushtaq, Muhammad Zia ur Rehman,
and Ernest Nlandu Kamavuako

_**learning (TL).**_

Manuscript received 12 July 2022; revised 24 August 2022;
accepted 24 August 2022. Date of publication 20 September 2022;
date of current version 31 October 2022. The associate editor
coordinating the review of this article and approving it for publication was Dr. Theerawit Wilaiprasitporn. (Corresponding author:
Muhammad Farrukh Qureshi.)

This work involved human subjects or animals in its research. Approval
of all ethical and experimental procedures and protocols was granted by
the Local Ethical Committee of Riphah International University under
Approval No. Riphah/RCRS/REC/000121/20012016, and performed in
line with the Declaration of Helsinki.

Muhammad Farrukh Qureshi and Zohaib Mushtaq are with the
Department of Electrical Engineering, Riphah International University, Islamabad 46000, Pakistan (e-mail: muhammad.farrukh@
riphah.edu.pk; zohaib.mushtaq@riphah.edu.pk).

Muhammad Zia ur Rehman is with the Department of Biomedical Engineering, Riphah International University, Islamabad 46000, Pakistan,
and also with NeXTlab, Università Campus Bio-Medico di Roma, 00128
Rome, Lazio, Italy (e-mail: ziaur.rehman@riphah.edu.pk).

Ernest Nlandu Kamavuako is with the Centre for Robotics Research,
Department of Engineering, King’s College London, WC2G 4BG
London, U.K. (e-mail: ernest.kamavuako@kcl.ac.uk).

Digital Object Identifier 10.1109/JSEN.2022.3204121

## I. INTRODUCTION

UMAN–COMPUTER interaction (HCI) applications
have drawn increasing interest in myoelectric control
H
devices in the past several decades [1], [2], [3]. The use of
hand movements to control peripheral devices is a characteristic shared by these systems. Electromyography (EMG)
has shown improved control and resilience compared with
other approaches [4], [5]. To classify EMG signals, several researchers have used classical machine learning techniques such as artificial neural networks [6], support vector
machines [7], [8], [9], [10], linear discriminant analysis [11],

[12], and _K_ -nearest neighbor [13], [14] for feature extraction.
However, feature extraction is time-consuming and requires
professional expertise to find the optimal feature set. Deep
neural network (DNN) addresses this problem by creating a
better representation from input data using multiple layers of
neural networks.

1558-1748 © 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:22 UTC from IEEE Xplore. Restrictions apply.

#### Source page 2

QURESHI et al.: SPECTRAL IMAGE-BASED MULTIDAY sEMG CLASSIFICATION OF HAND MOTIONS USING CNN FOR HCI 20677

TABLE I
ANALYSIS OF THE EXISTING STUDIES ON VARIOUS EMG DATASETS

DNN has been used in image classification [28], speech
recognition [29], sound classification [30], and video
classification [31]. One of the most famous DNN is the
convolutional neural network (CNN). In recent years, CNNbased surface EMG (sEMG) feature extraction improved hand
gesture recognition for short-term and long-term classification.
The CNN-based short-term classification (intrasession) has
been extensively researched. Ding _et_ _al._ [15] proposed a
parallel multiscale convolution architecture with bigger
kernel filter sizes. Tam _et_ _al._ [16] demonstrated a CNN
for real-time fine gesture detection for myoelectric control
on one healthy individual with 98.15% accuracy. In [17],
a hybrid architecture of the recurrent neural network (RNN)
and CNN was developed and the achieved accuracy was
87.2%, 94.1%, 99.7%, and 94.5%. Chen _et_ _al._ [18] evaluated
a compact CNN model using NinaPro DB5 and Myo dataset
with 70%–98.81% accuracy. Atzori _et_ _al._ [19] assessed an
average of 50 hand gestures in 67 able-bodied (healthy and
nondisabled) individuals and 11 transradial amputees using
CNN and obtained a classification accuracy between 38.09%
and 60.27%. In [20], a 3-D CNN with 3-D kernels was
used to gather the spatial and temporal characteristics from
successive sEMG images for an HD-sEMG-based gesture
recognition system and achieved 61.7%–98.6% accuracy.
Nasri _et_ _al._ [21] proposed Myo Armband gesture recognition.
They used a gated recurrent unit (GRU) network to train on
raw sEMG data and obtained 77.85% accuracy.

However, some works still provide results on long-term
classification (intersession/intersubject). Zhai _et_ _al._ [22] used
a self-recalibrating CNN that does not need user retraining.
The method was tested on 40 healthy and 11 amputee hands
in the NinaPro database. Wei _et_ _al._ [23] developed two-stream
CNN based on sEMG features for NinaPro and BioPatRec
with 94% and 80%–90% accuracy, respectively. Another
CNN-based feature extraction technique was developed in [24]
with 68% accuracy. In [25], a compact CNN architecture was
developed and it achieved an accuracy of 80.25%. A two-stage
RNN was deployed that achieved an accuracy between 34.8%
and 95% in [26]. In [27], a CNN was applied and it achieved a
classification accuracy of 87.7%–96.8%. Table I provides the
summary of these studies.

Several factors contribute to data variability in EMG such
as drift in sensor, sensor placement, and skin conductivity [26]. A long-term classification is necessary to mitigate
this issue. As seen in Table I, few of these research studies

use long-term classification. Therefore, this study incorporates
multiday data from [32] obtained from both healthy and
amputee patients. Furthermore, since a raw EMG signal is
1-D, stochastically distributed, and lacks spectral information,
it should be converted into a 2-D spectrogram before being fed
into a CNN. When input to CNN, a spectrogram provides a
better interpretation of spectrum information and demonstrates
improved processing. The following are the primary and
significant contributions of this experimental study.

1) Unique approach of Mel spectrogram images for classification of sEMG signals for upper limb is used.
2) Analysis of a proposed CNN on intrasession and intersession on the dataset for both the able-bodied (healthy
and nondisabled) and amputee subjects is done.
3) The performance of the proposed CNN is compared
with that of stacked sparse autoencoders (SSAEs) on
the given dataset in [32].
4) The same proposed CNN is compared with well-known
pretrained transfer learning (TL) weights such as
AlexNet, ResNet, and DenseNet.
The rest of this article is structured as follows. In Section II,
an introduction to the Mel spectrogram, preprocessing techniques, and the architecture of the proposed CNN are presented. The experimental results are presented in Section III.
Section IV provides further discussion of the results, and
Section V concludes this article.

## II. METHODOLOGY
The generalized block diagram is illustrated in
Fig. 1. The details of these blocks will be discussed in
Sections II-C and II-D.

### A. Mel-Spectrogram-Based Feature Extraction
Technique

Spectrogram is a Fourier transform of raw signal and
displays frequency content as a function of time and is
widely used for signal visualization. The Mel spectrogram is a
spectrogram that is divided into multiple points, each of which
distributes frequencies and times evenly on a Mel frequency
scale. In [30], the relationship is defined as

Mel = 2595 × log10

_f_
1 +
700

(1)

and its inverse is given as

_f_ = 700 ×

- Mel
10 2595 - 1 (2)

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:22 UTC from IEEE Xplore. Restrictions apply.

#### Source page 3

20678 IEEE SENSORS JOURNAL, VOL. 22, NO. 21, 1 NOVEMBER 2022

TABLE II
DETAILS OF THE USED DATASET

Fig. 2. Raw EMG single movement with four repetitions. The red signal
is the windowed signal with a length of 200 ms.

Fig. 1. Generalized block diagram of the work followed in this study.

where Mel is a Mel-based frequency, while the normal frequency is represented by _f_ .

### B. Experimental Datasets and Setup Description

In this article, a dataset from a previous work [32] is
being used. The data used included eight able-bodied (healthy
and nondisabled) volunteers and four transradial amputees.
Table II contains the demographic information on the subjects.
Three electrodes were connected with flexor carpi radialis,
palmaris longus muscle, and flexor digitorum superficialis,
while other three were connected with extensor carpi radialis
longus, extensor digitorum, and extensor carpi ulnaris and provided sEMG data as six channels. Each participant performed
11 hand motions: hand open, hand close, supination, pronation,
fine grasp, side grip, flex hand, extend hand, agree, pointer, and
rest. Each subject performed seven sessions separated by 24 h.
Each hand movement was repeated four times every session,
with a 5-s contraction and relaxation period. Fig. 2 illustrates
a recording of a single sEMG movement.

### C. Preprocessing Technique

An 8-kHz sample rate is used to obtain data from sEMG
signals. The signals are filtered at 10–500 Hz bandpass filter
and then 50 Hz notch filter. Each raw signal is divided into
200 ms windowed signals with an overlapping increment of
29 ms as shown in Fig. 2. There are 11 movements with four
repetitions; as a result, there are 400 windowed signals for

Fig. 3. (a) Windowed sEMG signal with six channels. Mel spectrograms
of the windowed signal with the parameter settings of (b) fmax = 128,
nfft = 1024, and hops = 16 and (c) fmax = 256, nfft = 1024, and
hops = 128.

each movement and 4400 windowed signals for each subject
in a single day. These signals become 30800 windowed signals
for each subject over a period of seven days.

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:22 UTC from IEEE Xplore. Restrictions apply.

#### Source page 4

QURESHI et al.: SPECTRAL IMAGE-BASED MULTIDAY sEMG CLASSIFICATION OF HAND MOTIONS USING CNN FOR HCI 20679

Fig. 4. Architecture of the five-layered proposed CNN.

Since there are six channels (corresponding to the number
of electrodes attached to subjects) in each windowed signal as
shown in Fig. 3(a), each windowed signal is aggregated vertically and then converted into image as Mel spectrograms for
each movement. The Mel spectrograms of the same signal with
different parameters’ settings are shown in Fig. 3(b) and (c).
Here, Mel frequency scales, number of samples per frame in
spectrogram, and time scale in the _x_ -axis are represented by

_f_ max, _n_ fft, and hops. We are using _f_ max = 256, _n_ fft = 1024,
and hops = 128 as depicted in Fig. 3(c). This conversion
resulted in 30800 images for each subject.

### D. CNN Architecture

The experimental process in this study involves a deep CNN
with the max-pooling function. The model is implemented
on the Mel spectrogram, and the input size of the feature is
224 × 224. The input is denoted as _I_, where _�_ is a parameter
of a composite nonlinear function denoted as _G(._ | _�)_ . The
operation of convolution can be described as [33]

_Y_ = _G(I_ | _�)_ = _gl(. . . g_ 3 _(g_ 2 _(I_ | _θ_ 2 _)_ | _θ_ 3 _)θL)_ (3)

where _gl_ is the _l_ th-layer of CNN, and _L_ denotes the total
number of layers to be used. In this case, _L_ = 5. The parameter
for the _l_ th-layer is _θ_ 1 = [ _X, b_ ]. Then the operations in the
convolutional layers can be described as

_Y_ 1 = _g_ 1 _(Il_ | _θL)_ = _h(Il_ + _b_ ∗ _X)_ (4)

where _Il_ is the input of the _l_ th-layer, _X_ denotes the corresponding filter, ∗ represents the valid convolution, _h(_ - _)_ denotes the
pointwise activation function, and _b_ denotes the vector bias
term.

The images were fed to CNN with the size of
224 × 224 × 3. A rescaling layer of range _(_ 0 _,_ 1 _)_ is also
introduced. In CNN, the _Adam_ optimizer [34] is used, the total
number of epochs is 50, the batch size is 32 uniformly, the
rectified linear unit (ReLU) activation function [35] is used for
the first four layers, and the Softmax activation function [36]
is used for the final layer. The general architecture of CNN
implementation is depicted in Fig. 4. The parameters of the
filters, the strided max-pooling function, and the receptive field
are described in the following.

1) _Layer_ _1:_ The model’s initial layer has 24 filters each
with a 9 × 9 receptive field. A batch normalization layer
is used to normalize the input for each mini batch. It is

### TABLE III
SUMMARY OF THE PROPOSED MODEL

followed by a 4 × 4 strided max-pooling function. ReLU
is used as the activation function.
2) _Layer_ _2:_ The second layer consists of 48 filters with a
receptive field of 5 × 5. Batch normalization layer is
applied. Following that, a 4 × 4 strided max-pooling is
performed using ReLU as the activation function.
3) _Layer_ _3:_ The third layer is composed of 48 filters with
a receptive field of 3 × 3. Padding is the “same” for
the feature extraction algorithms. Batch normalization is
applied and is followed by a 2 × 2 maximum pooling
using ReLU as the activation function.
4) _Layer 4:_ The fourth layer is the first fully connected (fc)
dense layer, consisting of 64 hidden units with ReLU as
its activation function.
5) _Layer_ _5:_ The last layer is the second fc dense layer.
It is made up of output units equal to the number of
classes in the dataset. In the last layer, the Softmax as
an activation function is applied.
The summary of the proposed model is given in Table III.

### E. Performance Evaluation Metrics

The performance evaluation metrics for multiclass classification such as accuracy, macro weighted precision (MWP),
macro weighted recall (MWR), and macro _F_ 1-score have been
used [37].

Authorized licensed use limited to: NUST School of Electrical Engineering and Computer Science (SEECS). Downloaded on June 24,2026 at 07:10:22 UTC from IEEE Xplore. Restrictions apply.

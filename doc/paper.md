---
title: 'GCA Analyzer: A Python Package for Group Conversation Analysis with Focus on Chinese Text'
tags:
  - Python
  - MOOC
  - Group Conversation Analysis
  - Group Dynamics
  - Text Processing
  - Social Interaction
  - Educational Research
authors:
  - name: [Jianjun Xiao]
    orcid: [0000-0003-0000-9630]
    affiliation: 1
affiliations:
 - name: [Beijing Normal University]
   index: 1
date: 2025-01-12
bibliography: paper.bib

---

# Summary

Group conversation analysis is crucial for understanding social dynamics, learning behaviors, and communication patterns in various settings, particularly in educational contexts. The GCA Analyzer is a Python package that implements a comprehensive set of metrics and methods for analyzing group conversations, with special emphasis on Chinese text processing capabilities. This tool provides quantitative measures for participation patterns, interaction dynamics, content novelty, and communication density, making it especially valuable for researchers in education, social psychology, and communication studies.

# Statement of Need

Understanding group conversation dynamics is essential in various fields, from educational research to organizational behavior studies. While several tools exist for conversation analysis, there is a notable gap in tools that can effectively handle text while providing comprehensive interaction metrics. The GCA Analyzer addresses this gap by providing:

1. Robust participation analysis through participation matrices
2. Temporal interaction analysis using sliding windows
3. Content similarity and novelty metrics
4. Social impact and responsivity measurements
5. Visualization capabilities for interaction patterns

These features enable researchers to conduct detailed analyses of group conversations, particularly in educational and organizational settings, supporting both research and practical applications.

# Mathematics

The GCA Analyzer implements several key mathematical formulas for analyzing group conversations:

## Participation Rate
For a participant $a$, the participation count $\|P_a\|$ and average participation rate $\bar{p_a}$ are calculated as:

$\|P_a\| = \sum_{t=1}^n M_{a,t}$

$\bar{p_a} = \frac{1}{n}\|P_a\|$

where $M_{a,t}$ is 1 if person $a$ contributes at time $t$, and 0 otherwise, and $n$ is the total number of contributions.

## Participation Standard Deviation
The participation standard deviation $\sigma_a$ for participant $a$ is:

$\sigma_a = \sqrt{\frac{1}{n-1}\sum_{t=1}^n (M_{a,t} - \bar{p_a})^2}$

## Normalized Participation Rate
The normalized participation rate ($\hat{P_a}$) is computed relative to equal participation:

$\hat{P_a} = \frac{\bar{p_a} - \frac{1}{k}}{\frac{1}{k}}$

where $k$ is the number of participants.

## Cross-Cohesion Matrix
The cross-cohesion matrix $\Xi$ for analyzing temporal interactions is computed as:

$\Xi_{ab} = \frac{1}{w} \sum_{\tau=1}^w \frac{\sum_{t \geq \tau} M_{a,t-\tau}M_{b,t}S_{t-\tau,t}}{\sum_{t \geq \tau} M_{a,t-\tau}M_{b,t}}$

where:
- $w$ is the optimal window length
- $S_{t-\tau,t}$ is the cosine similarity between messages at times $t-\tau$ and $t$
- $M_{a,t}$ and $M_{b,t}$ are participation indicators for persons $a$ and $b$ at time $t$

## Internal Cohesion
For each participant $a$, internal cohesion is their self-interaction:

$C_a = \Xi_{aa}$

## Overall Responsivity
The overall responsivity $R_a$ for participant $a$ is:

$R_a = \frac{1}{k-1}\sum_{b \neq a} \Xi_{ab}$

## Social Impact
The social impact $I_a$ for participant $a$ is:

$I_a = \frac{1}{k-1}\sum_{b \neq a} \Xi_{ba}$

## Message Newness
For a message $c_t$ at time $t$, its newness $n(c_t)$ is:

$n(c_t) = \frac{\|\text{proj}_{\perp H_t}(c_t)\|}{\|\text{proj}_{\perp H_t}(c_t)\| + \|c_t\|}$

where:
- $H_t$ is the space spanned by all previous message vectors
- $\text{proj}_{\perp H_t}$ is the orthogonal projection onto the complement of $H_t$
- $\|c_t\|$ is the norm of the current message vector

The overall newness $N_a$ for participant $a$ is:

$N_a = \frac{1}{\|P_a\|}\sum_{t \in T_a} n(c_t)$

where $T_a$ is the set of times when participant $a$ contributed.

## Communication Density
For a message $c_t$ at time $t$, its density $D_i$ is:

$D_i = \frac{\|c_t\|}{L_t}$

where $L_t$ is the word length of the message.

The average communication density $\bar{D_a}$ for participant $a$ is:

$\bar{D_a} = \frac{1}{\|P_a\|}\sum_{t \in T_a} D_i$

# References

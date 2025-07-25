Mathematical Foundation
========================

This section explains the mathematical principles behind GCA Analyzer's metrics.

Core Metrics
---------------

The GCA Analyzer implements three categories of metrics to provide comprehensive analysis of group communication dynamics:

1. **Participation Metrics**
   Measures the relative contribution frequency of each participant in the communication.  
   These metrics help identify participation patterns and balance within the group:  
   
   - Negative values indicate below-average participation  
   - Positive values indicate above-average participation  
   - Zero values represent perfectly balanced participation  

   The participation matrix M is defined as:

   .. math::

      M_{ij} = \begin{cases} 
      1 & \text{if person i participates at time j} \\
      0 & \text{otherwise}
      \end{cases}

   From this matrix, we derive several participation metrics:

   1. Raw Participation Rate (||Pa||), formula 4:

   .. math::

      ||Pa|| = \sum_{t=1}^{n} M_{a,t}

   2. Average Participation Rate (p̄a), formula 5:

   .. math::

      \bar{p}_a = \frac{1}{n}||Pa||

   3. Adjusted Participation Rate (P̂a), formula 9:

   .. math::

      \hat{P}_a = \frac{\bar{p}_a - 1/k}{1/k}

   where k is the number of participants and n is the total number of contributions.

2. **Interaction Metrics**
   Analyzes how participants interact with each other through three key aspects:

   - **Responsivity**: Measures how well participants respond to others' contributions  
   - **Internal Cohesion**: Evaluates the consistency and coherence of individual contributions  
   - **Social Impact**: Quantifies each participant's influence on the group communication  

   The interaction metrics are based on cross-cohesion analysis:

   1. Cross-Cohesion Matrix (Ξ), formula 17:

   .. math::

      \Xi_{ab} = \frac{1}{w}\sum_{\tau=1}^{w}\frac{\sum_{t=\tau+1}^{n}M_{a,t-\tau}M_{b,t}S_{t-\tau,t}}{\sum_{t=\tau+1}^{n}M_{a,t-\tau}M_{b,t}}

   where:
   
   - w is the optimal window size
   - S_{t-τ,t} is the semantic similarity between contributions at t-τ and t

   2. Internal Cohesion (Ca), formula 18:

   .. math::

      C_a = \Xi_{aa}

   3. Overall Responsivity (Ra), formula 19:

   .. math::

      R_a = \frac{1}{k-1}\sum_{b \neq a}\Xi_{ab}

   4. Social Impact (Ia), formula 20:

   .. math::

      I_a = \frac{1}{k-1}\sum_{b \neq a}\Xi_{ba}

3. **Content Analysis Metrics**
   Evaluates the semantic aspects of contributions using advanced NLP techniques:

   - **Content Newness**: Measures the introduction of novel content to the communication  
   - **Communication Density**: Assesses the information density of contributions  

   Content analysis is performed using LSA (Latent Semantic Analysis). For any given contribution, 
   its semantic content can be decomposed into two parts: information that is already given in 
   previous discourse and new information being introduced.

   1. Content Newness (n(ci)), formula 25:
   
   First, we define the given subspace Gi for contribution i as the span of all previous document vectors:

   .. math::

      G_i = span\{\vec{d_1}, \vec{d_2}, ..., \vec{d_{i-1}}\}

   The semantic content can then be decomposed into:
   
   - Given content: projection onto the given subspace
   
   .. math::
   
      \vec{g_i} = Proj_{G_i}(\vec{d_i})
   
   - New content: projection onto orthogonal complement
   
   .. math::
   
      \vec{n_i} = Proj_{G_i^\perp}(\vec{d_i})
   
   The total semantic content is completely partitioned by these projections:
   
   .. math::
   
      \vec{d_i} = \vec{g_i} + \vec{n_i}

   The content newness measure is then calculated as:

   .. math::

      n(c_i) = \frac{||\vec{n_i}||}{||\vec{n_i}|| + ||\vec{g_i}||}

   where:

   - \vec{n_i} is the new content vector (perpendicular to given subspace)
   - \vec{g_i} is the given content vector (projection onto given subspace)
   
   This given-new value ranges from:
   
   - 0: all given content (nothing new)
   - 1: all new content (completely novel)
   
   For a participant's overall newness score, we average over all their contributions:
   
   .. math::
   
      N_a = \frac{1}{|P_a|}\sum_{i \in P_a}n(c_i)
   
   where Pa is the set of all contributions by participant a.

   2. Communication Density (Di), formula 27:

   .. math::

      D_i = \frac{||c_t||}{L_t}

   where:  

   - ||c_t|| is the norm of the contribution vector  
   - L_t is the length of the text message  

Window Size Optimization
-------------------------

The optimal window size w* is determined by:

.. math::

   w* = \argmax_{w \in [w_{min}, w_{max}]} P(w)

where:  

- P(w) is the proportion of active participants in window w  
- w_{min} and w_{max} are configurable minimum and maximum window sizes  

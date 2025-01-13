Mathematical Foundation
========================

This section explains the mathematical principles behind GCA Analyzer's metrics.

Core Metrics
---------------

The GCA Analyzer implements three categories of metrics to provide comprehensive analysis of group conversation dynamics:

1. **Participation Metrics**
   Measures the relative contribution frequency of each participant in the conversation.  
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

   1. Raw Participation Rate (||Pa||):

   .. math::

      ||Pa|| = \sum_{t=1}^{n} M_{a,t}

   2. Average Participation Rate (p̄a):

   .. math::

      \bar{p}_a = \frac{1}{n}||Pa||

   3. Participation Standard Deviation (σa):

   .. math::

      \sigma_a = \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(M_{a,t} - \bar{p}_a)^2}

   4. Normalized Participation Rate (P̂a):

   .. math::

      \hat{P}_a = \frac{\bar{p}_a - 1/k}{1/k}

   where k is the number of participants and n is the total number of contributions.

2. **Interaction Metrics**
   Analyzes how participants interact with each other through three key aspects:

   - **Responsivity**: Measures how well participants respond to others' contributions  
   - **Internal Cohesion**: Evaluates the consistency and coherence of individual contributions  
   - **Social Impact**: Quantifies each participant's influence on the group discussion  

   The interaction metrics are based on cross-cohesion analysis:

   1. Cross-Cohesion Matrix (Ξ):

   .. math::

      \Xi_{ab} = \frac{1}{w}\sum_{\tau=1}^{w}\frac{\sum_{t=\tau+1}^{n}M_{a,t-\tau}M_{b,t}S_{t-\tau,t}}{\sum_{t=\tau+1}^{n}M_{a,t-\tau}M_{b,t}}

   where:
   - w is the optimal window size
   - S_{t-τ,t} is the semantic similarity between contributions at t-τ and t

   2. Internal Cohesion (Ca):

   .. math::

      C_a = \Xi_{aa}

   3. Overall Responsivity (Ra):

   .. math::

      R_a = \frac{1}{k-1}\sum_{b \neq a}\Xi_{ab}

   4. Social Impact (Ia):

   .. math::

      I_a = \frac{1}{k-1}\sum_{b \neq a}\Xi_{ba}

3. **Content Analysis Metrics**
   Evaluates the semantic aspects of contributions using advanced NLP techniques:

   - **Message Newness**: Measures the introduction of novel content to the discussion  
   - **Communication Density**: Assesses the information density of contributions  

   Content analysis is performed using LSA (Latent Semantic Analysis):

   1. Message Newness (n(ct)):

   .. math::

      n(c_t) = \frac{||proj_{\perp H_t}(c_t)||}{||proj_{\perp H_t}(c_t)|| + ||c_t||}

   where:
   - ct is the current contribution vector  
   - H_t is the subspace spanned by previous contributions  

   2. Communication Density (Di):

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

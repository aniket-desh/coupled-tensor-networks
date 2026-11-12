# Coupled tensor networks

![Eight-leaf binary tree tensor network](figures/ttn8_diagram.png)

```math
f_o(x) = \sum_{\alpha,\beta,q,a,i} C_{o\alpha\beta}\,(W_{\rm L})_{\alpha q_1 q_2}\,(W_{\rm R})_{\beta q_3 q_4}
\prod_{k=1}^{4} (W_k)_{q_k a_{2k-1} a_{2k}} \prod_{j=1}^{8} (A_j)_{a_j i_j}\,\phi(x^{(j)})_{i_j}
```

Shared-feature model and its two-component teacher, with $\rho = s_2/s_1$:

```math
T_{oij} = u_i\,(RB)_{oj}, \qquad
T^\star = s_1\, e_1 \otimes g \otimes h_1 + s_2\, e_2 \otimes g \otimes h_2, \qquad
c_k = \langle T,\ e_k \otimes g \otimes h_k \rangle
```

Hierarchy model and its four-component teacher, with groups $G = \{1,2\}, \{3,4\}$:

```math
T_{oijl} = u_i\, F_{gj}\, P_{ag}\, R_{oab}\, B_{bl}, \qquad
T^\star = \sum_{k=1}^{4} s_k\, e_k \otimes g \otimes h_{G(k)} \otimes w_k
```

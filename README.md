# Rough example representing Kolmogrov-Arnold Network (KAN) as a Block Diagonal Multilayer-Perceptron Network (MLP)

Format from [awesome-kan](https://github.com/mintisan/awesome-kan) : Awesome KAN(Kolmogorov-Arnold Network) ｜ ![Github stars](https://img.shields.io/github/stars/mintisan/awesome-kan.svg))

Examples are based on [FastKAN](https://github.com/ZiyaoLi/fast-kan) : Very Fast Calculation of Kolmogorov-Arnold Networks (KAN)  ｜ ![Github stars](https://img.shields.io/github/stars/ZiyaoLi/fast-kan.svg)

I used the Gaussian as the activation function, which I think reduces to a Gaussian Radial Basis Function KAN with variable grid points. See [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) : Kolmogorov-Arnold Networks (KAN) using Chebyshev polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/SynodicMonth/ChebyKAN.svg)

See discussions:
- [[D] Kolmogorov-Arnold Network is just an MLP](https://www.reddit.com/r/MachineLearning/comments/1clcu5i/d_kolmogorovarnold_network_is_just_an_mlp/)
- [[D] Kolmogorov-Arnold Network is just an MLP (Twitter) ](https://x.com/bozavlado/status/1787376558484709691)

Example for one layer, 2 inputs, 3 basis curves, and 3 outputs. For each output, you need to stack the block diagonal matrices.

$$ 
W_1 = \begin{bmatrix}
a_1 & 0 \\
a_2 & 0 \\
a_3 & 0 \\
0   & a_4 \\
0   & a_5 \\
0   & a_6 \\
a_7 & 0 \\
a_8 & 0 \\
a_9 & 0 \\
0   & a_{10} \\
0   & a_{11} \\
0   & a_{12} \\
a_{13} & 0 \\
a_{14} & 0 \\
a_{15} & 0 \\
0   & a_{16} \\
0   & a_{17} \\
0   & a_{18} \\
\end{bmatrix}
$$

$$
v = \begin{bmatrix}
x \\
y
\end{bmatrix}
$$

The first linear layer output is then

$$
\boldsymbol{z_1} = W_1v =
\begin{bmatrix}
a_1 x\\
a_2 x\\
a_3 x\\
a_4 y\\
a_5 y\\
a_6 y\\
a_7 x\\
a_8 x\\
a_9 x\\
a_{10} y\\
a_{11} y\\
a_{12} y\\
a_{13} x\\
a_{14} x\\
a_{15} x\\
a_{16} y\\
a_{17} y\\
a_{18} y\\
\end{bmatrix}
$$

Apply activation function $\sigma$ and bias vector $b \in \mathbb{R}^{18}$ to get $\boldsymbol{z_2} = \sigma (\boldsymbol{z_1} + b)$.

$$ 
W_2 = \begin{bmatrix}
\color{red}{c_1} & \color{red}{c_2} & \color{red}{c_3} & \color{blue}{c_4} & \color{blue}{c_5} & \color{blue}{c_6} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & \color{red}{c_7} & \color{red}{c_8} & \color{red}{c_9} & \color{blue}{c_{10}} & \color{blue}{c_{11}} & \color{blue}{c_{12}} & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0  & \color{red}{c_{13}} & \color{red}{c_{14}} & \color{red}{c_{15}} & \color{blue}{c_{16}} & \color{blue}{c_{17}} & \color{blue}{c_{18}} \\
\end{bmatrix}
$$

Lastly, do $\boldsymbol{z_3} = W_2 z_3$ to get the output. If you only care about the red parameters in $W_2$, you get the curves for each input $x$, and vice versa for interpretability. Choose $\sigma(x) = e^{-x^2}$, then you have a gaussian radial basis function interpolation with variable grid points. Note that the gaussian basis is similar to the B-spline basis depicted in the original KAN paper.



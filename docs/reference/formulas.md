# Formula catalog

Generated from the repository by `python scripts/generate_reference.py`.

**285 formulas across 10 domains.**

Every name below is available through `MathForge().get(name)`. Formulas are symbolic building blocks; check their assumptions and input domains before interpreting a result. A symbolic optimization step is separate from a model optimizer, and symbolic metric expressions are separate from the batch evaluation API.

```python
from neurogebra import MathForge

expression = MathForge().get("swish")
print(expression.symbolic_expr)
print(expression.params)
print(expression.gradient("x").symbolic_expr)
```

## Activations (15)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `elu` | `Piecewise((x, x > 0), (alpha*(exp(x) - 1), True))` | alpha=1.0 |
| `gelu` | `0.5*x*(tanh(sqrt(2)*(0.044715*x**3 + x)/sqrt(pi)) + 1)` | None |
| `hard_sigmoid` | `Max(0, Min(1, x/6 + 1/2))` | None |
| `hard_swish` | `x*Max(0, Min(1, x/6 + 1/2))` | None |
| `leaky_relu` | `Max(x, alpha*x)` | alpha=0.01 |
| `linear` | `x` | None |
| `mish` | `x*tanh(log(exp(x) + 1))` | None |
| `relu` | `Max(0, x)` | None |
| `selu` | `1.0507009873554805*Piecewise((x, x > 0), (1.6732632423543772*exp(x) - 1.6732632423543772, True))` | None |
| `sigmoid` | `1/(1 + exp(-x))` | None |
| `softplus` | `log(exp(x) + 1)` | None |
| `softsign` | `x/(Abs(x) + 1)` | None |
| `square` | `x**2` | None |
| `swish` | `x/(1 + exp(-x))` | None |
| `tanh` | `tanh(x)` | None |

## Losses (8)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `binary_crossentropy` | `-y_true*log(y_pred) - (1 - y_true)*log(1 - y_pred)` | None |
| `hinge` | `Max(0, -y_pred*y_true + 1)` | None |
| `huber` | `Piecewise((0.5*(y_pred - y_true)**2, delta >= Abs(y_pred - y_true)), (-0.5*delta**2 + delta*Abs(y_pred - y_true), True))` | delta=1.0 |
| `log_cosh` | `log(cosh(y_pred - y_true))` | None |
| `mae` | `Abs(y_pred - y_true)` | None |
| `mse` | `(y_pred - y_true)**2` | None |
| `quantile` | `Piecewise((q*(-y_pred + y_true), y_pred <= y_true), ((1 - q)*(y_pred - y_true), True))` | q=0.5 |
| `squared_hinge` | `Max(0, -y_pred*y_true + 1)**2` | None |

## Regularizers (20)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `cauchy_reg` | `lambda_reg*log(1 + w**2/sigma**2)` | lambda_reg=0.01, sigma=1.0 |
| `confidence_penalty` | `-lambda_reg*p*log(epsilon + p)` | lambda_reg=0.1, epsilon=1e-08 |
| `elastic_net` | `alpha*lambda_reg*Abs(w) + lambda_reg*w**2*(1 - alpha)` | lambda_reg=0.01, alpha=0.5 |
| `entropy_reg` | `-lambda_reg*(p*log(epsilon + p) + (1 - p)*log(epsilon - p + 1))` | lambda_reg=0.01, epsilon=1e-08 |
| `gradient_penalty` | `lambda_reg*(grad_norm - 1)**2` | lambda_reg=10.0 |
| `group_lasso` | `lambda_reg*sqrt(epsilon + w1**2 + w2**2)` | lambda_reg=0.01, epsilon=1e-08 |
| `kl_divergence_reg` | `lambda_reg*(p*log((epsilon + p)/(epsilon + q)) + (1 - p)*log((epsilon - p + 1)/(epsilon - q + 1)))` | lambda_reg=0.01, epsilon=1e-08 |
| `l1` | `lambda_reg*Abs(w)` | lambda_reg=0.01 |
| `l1_l2` | `l1_reg*Abs(w) + l2_reg*w**2` | l1_reg=0.01, l2_reg=0.01 |
| `l2` | `lambda_reg*w**2` | lambda_reg=0.01 |
| `label_smoothing` | `y*(1 - epsilon_smooth) + epsilon_smooth/K` | epsilon_smooth=0.1, K=10 |
| `log_barrier` | `-lambda_reg*log(epsilon - w**2 + 1)` | lambda_reg=0.01, epsilon=1e-08 |
| `max_norm` | `Max(0, -max_val**2 + w**2)` | max_val=3.0 |
| `mcp` | `Piecewise((lambda_reg*Abs(w) - w**2/(2*gamma_mcp), Abs(w) <= gamma_mcp*lambda_reg), (gamma_mcp*lambda_reg**2/2, True))` | lambda_reg=0.01, gamma_mcp=3.0 |
| `orthogonal_reg` | `lambda_reg*w1**2*w2**2` | lambda_reg=0.01 |
| `scad` | `Piecewise((lambda_reg*Abs(w), lambda_reg >= Abs(w)), ((2*a*lambda_reg*Abs(w) - lambda_reg**2 - w**2)/(2*a - 2), Abs(w) <= a*lambda_reg), (lambda_reg**2*(a + 1)/2, True))` | lambda_reg=0.01, a=3.7 |
| `sqrt_reg` | `lambda_reg*sqrt(epsilon + w**2)` | lambda_reg=0.01, epsilon=1e-08 |
| `tikhonov` | `lambda_reg*(w - w_prior)**2` | lambda_reg=0.01, w_prior=0.0 |
| `total_variation` | `lambda_reg*Abs(w_i - w_j)` | lambda_reg=0.01 |
| `weight_decay` | `0.5*lambda_reg*w**2` | lambda_reg=0.01 |

## Algebra (48)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `absolute_value` | `Abs(x)` | None |
| `beta_distribution` | `x**(alpha - 1)*(1 - x)**(beta_param - 1)` | alpha=2.0, beta_param=5.0 |
| `cauchy_distribution` | `1/(pi*gamma_param*(1 + (x - x0)**2/gamma_param**2))` | x0=0.0, gamma_param=1.0 |
| `clamp` | `Max(low, Min(high, x))` | low=0.0, high=1.0 |
| `cubic` | `a*x**3 + b*x**2 + c*x + d` | a=1, b=0, c=0, d=0 |
| `damped_oscillation` | `A*exp(-gamma_d*x)*cos(omega*x + phi)` | A=1.0, gamma_d=0.1, omega=1.0, phi=0.0 |
| `double_exponential` | `A*exp(-Abs(mu - x)/b)` | A=1.0, mu=0.0, b=1.0 |
| `exp_decay` | `A*exp(-k*x)` | A=1.0, k=1.0 |
| `exp_growth` | `A*exp(k*x)` | A=1.0, k=0.1 |
| `fourier_term` | `a0 + a1*cos(x) + a2*cos(2*x) + b1*sin(x) + b2*sin(2*x)` | a0=0.0, a1=1.0, b1=0.0, a2=0.0, b2=0.0 |
| `gaussian` | `A*exp(-(-mu + x)**2/(2*sigma**2))` | A=1.0, mu=0.0, sigma=1.0 |
| `gompertz` | `a*exp(-b*exp(-c*x))` | a=1.0, b=1.0, c=1.0 |
| `heaviside` | `Piecewise((0, x < 0), (1, True))` | None |
| `hill_equation` | `V_max*x**n/(K**n + x**n)` | V_max=1.0, K=0.5, n=2.0 |
| `inverse_square` | `a/(epsilon + x**2)` | a=1.0, epsilon=1e-08 |
| `laplace_distribution` | `exp(-Abs(mu - x)/b)/(2*b)` | mu=0.0, b=1.0 |
| `laplacian_kernel` | `exp(-gamma_k*Abs(x - y))` | gamma_k=1.0 |
| `lerp` | `a_val + t*(-a_val + b_val)` | a_val=0.0, b_val=1.0 |
| `linear_eq` | `b + m*x` | m=1, b=0 |
| `log_linear` | `a*log(exp(b*x) + 1)` | a=1.0, b=1.0 |
| `logarithmic` | `a*log(epsilon + x) + b` | a=1.0, b=0.0, epsilon=1e-08 |
| `logistic` | `L/(1 + exp(-k*(x - x0)))` | L=1.0, k=1.0, x0=0.0 |
| `lorentzian` | `A/(1 + (x - x0)**2/gamma_l**2)` | A=1.0, x0=0.0, gamma_l=1.0 |
| `matern_12_kernel` | `sigma_k**2*exp(-Abs(x - y)/length_scale)` | sigma_k=1.0, length_scale=1.0 |
| `matern_32_kernel` | `sigma_k**2*(1 + sqrt(3)*Abs(x - y)/length_scale)*exp(-sqrt(3)*Abs(x - y)/length_scale)` | sigma_k=1.0, length_scale=1.0 |
| `michaelis_menten` | `V_max*x/(K_m + x)` | V_max=1.0, K_m=0.5 |
| `monomial` | `a*x**n` | a=1.0, n=2.0 |
| `nth_root` | `(epsilon + Abs(x))**(1/n)` | n=3.0, epsilon=1e-08 |
| `periodic_kernel` | `sigma_k**2*exp(-2*sin(pi*Abs(x - y)/period)**2/length_scale**2)` | sigma_k=1.0, length_scale=1.0, period=1.0 |
| `polynomial_kernel` | `(alpha_k*x*y + c_k)**d_k` | alpha_k=1.0, c_k=1.0, d_k=3.0 |
| `power_law` | `a*x**n` | a=1.0, n=2.0 |
| `probit` | `0.5*erf(sqrt(2)*x/2) + 0.5` | None |
| `quadratic` | `a*x**2 + b*x + c` | a=1, b=0, c=0 |
| `quartic` | `a*x**4 + b*x**3 + c*x**2 + d*x + e` | a=1, b=0, c=0, d=0, e=0 |
| `ramp` | `Max(0, x)` | None |
| `rational` | `(a*x + b)/(c*x + d + epsilon)` | a=1.0, b=0.0, c=0.0, d=1.0, epsilon=1e-08 |
| `rational_quadratic_kernel` | `(1 + (x - y)**2/(2*alpha_rq*length_scale**2))**(-alpha_rq)` | alpha_rq=1.0, length_scale=1.0 |
| `rayleigh` | `x*exp(-x**2/(2*sigma**2))/sigma**2` | sigma=1.0 |
| `rbf_kernel` | `exp(-gamma_k*(x - y)**2)` | gamma_k=1.0 |
| `richards_curve` | `K/(1 + exp(-r*(x - x0)))**(1/nu)` | K=1.0, r=1.0, x0=0.0, nu=1.0 |
| `sawtooth_approx` | `-(sin(x) + sin(2*x)/2 + sin(3*x)/3)/pi + 0.5` | None |
| `sign_function` | `Piecewise((-1, x < 0), (0, Eq(x, 0)), (1, True))` | None |
| `sinusoidal` | `A*sin(omega*x + phi)` | A=1.0, omega=1.0, phi=0.0 |
| `smootherstep` | `6*t**5 - 15*t**4 + 10*t**3` | None |
| `smoothstep` | `-2*t**3 + 3*t**2` | None |
| `square_root` | `a*sqrt(epsilon + Abs(x))` | a=1.0, epsilon=1e-08 |
| `square_wave_approx` | `4*(sin(x) + sin(3*x)/3 + sin(5*x)/5)/pi` | None |
| `student_t` | `(1 + x**2/nu)**(-nu/2 - 1/2)*gamma_func(nu/2 + 1/2)/(sqrt(pi)*sqrt(nu)*gamma_func(nu/2))` | nu=5.0 |

## Calculus (48)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `abs_val` | `Abs(x)` | None |
| `arccos` | `acos(x)` | None |
| `arccosh` | `acosh(x)` | None |
| `arcsin` | `asin(x)` | None |
| `arcsinh` | `asinh(x)` | None |
| `arctan` | `atan(x)` | None |
| `arctan2` | `atan2(y, x)` | None |
| `arctanh` | `atanh(x)` | None |
| `beta_func` | `gamma(a_param)*gamma(b_param)/gamma(a_param + b_param)` | a_param=1.0, b_param=1.0 |
| `cbrt` | `x**(1/3)` | None |
| `chain_rule` | `df_du*du_dx` | None |
| `cos` | `cos(x)` | None |
| `cosh` | `cosh(x)` | None |
| `cot` | `cos(x)/sin(x)` | None |
| `csc` | `1/sin(x)` | None |
| `digamma` | `polygamma(0, x)` | None |
| `erf` | `erf(x)` | None |
| `erfc` | `erfc(x)` | None |
| `exp` | `exp(x)` | None |
| `finite_difference` | `(-f_minus + f_plus)/(2*h)` | h=1e-05 |
| `fourier_kernel_imag` | `-sin(omega*t)` | None |
| `fourier_kernel_real` | `cos(omega*t)` | None |
| `gamma_func` | `gamma(x)` | None |
| `gradient_descent_step` | `-gradient*lr + w` | lr=0.01 |
| `laplace_kernel` | `exp(-s*t)` | None |
| `ln` | `log(x)` | None |
| `log10` | `log(x)/log(10)` | None |
| `log2` | `log(x)/log(2)` | None |
| `product_rule` | `df_dx*g_val + dg_dx*f_val` | None |
| `quotient_rule` | `(df_dx*g_val - dg_dx*f_val)/(epsilon + g_val**2)` | epsilon=1e-08 |
| `reciprocal` | `1/x` | None |
| `sec` | `1/cos(x)` | None |
| `sech` | `1/cosh(x)` | None |
| `simpsons_rule` | `h*(f_a + f_b + 4*f_m)/6` | None |
| `sin` | `sin(x)` | None |
| `sinc` | `Piecewise((1, Eq(x, 0)), (sin(pi*x)/(pi*x), True))` | None |
| `sinh` | `sinh(x)` | None |
| `sqrt` | `sqrt(x)` | None |
| `tan` | `tan(x)` | None |
| `tanh_func` | `tanh(x)` | None |
| `taylor_arctan` | `-x**7/7 + x**5/5 - x**3/3 + x` | None |
| `taylor_cos` | `-x**6/720 + x**4/24 - x**2/2 + 1` | None |
| `taylor_exp` | `x**5/120 + x**4/24 + x**3/6 + x**2/2 + x + 1` | None |
| `taylor_ln1px` | `x**5/5 - x**4/4 + x**3/3 - x**2/2 + x` | None |
| `taylor_sin` | `-x**7/5040 + x**5/120 - x**3/6 + x` | None |
| `trapezoidal_rule` | `h*(f_a + f_b)/2` | None |
| `wavelet_mexican_hat` | `2*sqrt(3)*(1 - x**2)*exp(-x**2/2)/(3*pi**(1/4))` | None |
| `wavelet_morlet` | `exp(-x**2/2)*cos(5*x)` | None |

## Statistics (35)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `bayes_posterior_kernel` | `exp(-(-x + y)**2/(2*sigma_lik**2))*exp(-(-mu_prior + x)**2/(2*sigma_prior**2))` | mu_prior=0, sigma_prior=1, sigma_lik=1 |
| `beta_pdf` | `x**(alpha - 1)*(1 - x)**(beta_param - 1)` | alpha=2, beta_param=5 |
| `binary_entropy` | `-p*log(p + 1.0e-15) - (1 - p)*log(1.0 - p)` | None |
| `cauchy_pdf` | `1/(pi*gamma_param*(1 + (x - x0)**2/gamma_param**2))` | x0=0, gamma_param=1 |
| `characteristic_normal` | `exp(I*mu*t - sigma**2*t**2/2)` | mu=0, sigma=1 |
| `chi_squared_kernel` | `x**(k/2 - 1)*exp(-x/2)` | k=2 |
| `coefficient_of_variation` | `sigma/mu` | mu=1, sigma=0.1 |
| `cross_entropy_elem` | `-y*log(p + 1.0e-15) - (1 - y)*log(1.0 - p)` | None |
| `evidence_lower_bound` | `-kl_term + log_likelihood` | log_likelihood=0, kl_term=0 |
| `exponential_cdf` | `1 - exp(-lambda_param*x)` | lambda_param=1 |
| `exponential_pdf` | `lambda_param*exp(-lambda_param*x)` | lambda_param=1 |
| `gumbel_pdf` | `exp(-z - exp(-z))/beta_param` | mu=0, beta_param=1 |
| `js_divergence_elem` | `p*log(2*p/(p + q + 1.0e-15))/2 + q*log(2*q/(p + q + 1.0e-15))/2` | None |
| `kl_divergence_elem` | `p*log((p + 1.0e-15)/(q + 1.0e-15))` | None |
| `laplace_pdf` | `exp(-Abs(mu - x)/b)/(2*b)` | mu=0, b=1 |
| `linear_regression_pred` | `beta_0 + beta_1*x` | beta_0=0, beta_1=1 |
| `log_normal_pdf` | `sqrt(2)*exp(-(-mu + log(x))**2/(2*sigma**2))/(2*sqrt(pi)*sigma*x)` | mu=0, sigma=1 |
| `log_prior_normal` | `-log(sigma) - log(2*pi)/2 - (-mu + x)**2/(2*sigma**2)` | mu=0, sigma=1 |
| `logistic_cdf` | `1/(exp((mu - x)/s) + 1)` | mu=0, s=1 |
| `logistic_regression_pred` | `1/(exp(-beta_0 - beta_1*x) + 1)` | beta_0=0, beta_1=1 |
| `mgf_exponential` | `lambda_param/(lambda_param - t)` | lambda_param=1 |
| `mgf_normal` | `exp(mu*t + sigma**2*t**2/2)` | mu=0, sigma=1 |
| `mutual_info_bound` | `log(exp(x) + 1)` | None |
| `normal_pdf` | `sqrt(2)*exp(-(-mu + x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)` | mu=0, sigma=1 |
| `pearson_r_component` | `(-mu_x + x)*(-mu_y + y)/(sigma_x*sigma_y)` | mu_x=0, mu_y=0, sigma_x=1, sigma_y=1 |
| `pooled_variance` | `(s1**2*(n1 - 1) + s2**2*(n2 - 1))/(n1 + n2 - 2)` | n1=30, s1=1, n2=30, s2=1 |
| `rayleigh_pdf` | `x*exp(-x**2/(2*sigma**2))/sigma**2` | sigma=1 |
| `standard_error` | `sigma/sqrt(n)` | sigma=1, n=30 |
| `standard_normal_pdf` | `sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))` | None |
| `student_t_kernel` | `(1 + x**2/nu)**(-nu/2 - 1/2)` | nu=3 |
| `survival_exponential` | `exp(-lambda_param*x)` | lambda_param=1 |
| `t_statistic` | `sqrt(n)*(-mu + x)/s` | mu=0, s=1, n=30 |
| `uniform_pdf` | `1/(-a + b)` | a=0, b=1 |
| `weibull_survival` | `exp(-(x/lambda_param)**k)` | lambda_param=1, k=1.5 |
| `z_score` | `(-mu + x)/sigma` | mu=0, sigma=1 |

## Linear algebra (24)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `batch_norm_elem` | `beta_bn + gamma_bn*(-mu_batch + x)/sqrt(var_batch + 1.0e-5)` | mu_batch=0, var_batch=1, gamma_bn=1, beta_bn=0 |
| `canberra_dist_elem` | `Abs(x - y)/(Abs(x) + Abs(y) + 1.0e-8)` | None |
| `chebyshev_dist` | `Max(Abs(a - b), Abs(x - y))` | None |
| `cosine_similarity` | `dot_xy/(norm_x*norm_y + 1.0e-8)` | dot_xy=1, norm_x=1, norm_y=1 |
| `determinant_2x2` | `a*d - b*c` | None |
| `dot_product_elem` | `x*y` | None |
| `eigenvalue_2x2` | `a/2 + d/2 + sqrt(b*c + (a/2 - d/2)**2)` | None |
| `euclidean_dist_elem` | `(x - y)**2` | None |
| `frobenius_elem` | `x**2` | None |
| `huber_norm_elem` | `Piecewise((x**2/2, delta >= Abs(x)), (delta*(-delta/2 + Abs(x)), True))` | delta=1 |
| `l1_norm_elem` | `Abs(x)` | None |
| `l2_norm_elem` | `x**2` | None |
| `layer_norm_elem` | `beta_ln + gamma_ln*(-mu + x)/(sigma + 1.0e-5)` | mu=0, sigma=1, gamma_ln=1, beta_ln=0 |
| `log_softmax_score` | `x - log(exp(x) + exp(y) + 1.0e-8)` | None |
| `lp_norm_elem` | `Abs(x)**p` | p=2 |
| `mahalanobis_1d` | `Abs(mu - x)/sigma` | mu=0, sigma=1 |
| `manhattan_dist_elem` | `Abs(x - y)` | None |
| `minkowski_dist_elem` | `Abs(x - y)**p` | p=2 |
| `orthogonal_residual_elem` | `a - b*proj_coeff` | proj_coeff=0 |
| `scaled_dot_product` | `dot_qk/sqrt(d_k)` | dot_qk=1, d_k=64 |
| `sigmoid_kernel` | `tanh(alpha*x*y + c_coeff)` | alpha=1, c_coeff=0 |
| `softmax_score` | `exp(x)/(exp(x) + exp(y) + 1.0e-8)` | None |
| `trace_2x2` | `a + d` | None |
| `vector_projection_scalar` | `dot_ab/(norm_b**2 + 1.0e-8)` | dot_ab=1, norm_b=1 |

## Optimization (27)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `ackley_2d` | `-exp(0.5*cos(2*pi*x) + 0.5*cos(2*pi*y)) + E + 20 - 20*exp(-0.2*sqrt(0.5*x**2 + 0.5*y**2))` | None |
| `adagrad_step` | `-grad*lr/sqrt(G + 1.0e-8) + w` | lr=0.01, G=0 |
| `adam_bias_correction_m` | `m/(1 - beta1**t)` | beta1=0.9, m=0, t=1 |
| `adam_bias_correction_v` | `v_adam/(1 - beta2**t)` | beta2=0.999, v_adam=0, t=1 |
| `adam_m_update` | `beta1*m + grad*(1 - beta1)` | beta1=0.9, m=0 |
| `adam_step` | `-lr*m_hat/(sqrt(v_hat) + 1.0e-8) + w` | lr=0.001, m_hat=0, v_hat=0 |
| `adam_v_update` | `beta2*v_adam + grad**2*(1 - beta2)` | beta2=0.999, v_adam=0 |
| `adamw_step` | `-lr*m_hat/(sqrt(v_hat) + 1.0e-8) + w*(-lambda_wd*lr + 1)` | lr=0.001, m_hat=0, v_hat=0, lambda_wd=0.01 |
| `constant_lr` | `lr_init` | lr_init=0.001 |
| `cosine_annealing_lr` | `lr_min + (lr_max - lr_min)*(cos(pi*epoch/T_max) + 1)/2` | lr_min=1e-06, lr_max=0.01, T_max=100 |
| `cyclical_lr` | `lr_min + (lr_max - lr_min)*Abs(epoch/step_size - 2*floor(epoch/(2*step_size) + 1/2))` | lr_min=0.0001, lr_max=0.01, step_size=2000 |
| `ema_update` | `alpha_ema*param_new + param_ema*(1 - alpha_ema)` | alpha_ema=0.001 |
| `exponential_decay_lr` | `lr_init*exp(-epoch*k_decay)` | lr_init=0.01, k_decay=0.01 |
| `gradient_clip_norm` | `grad*Min(1, max_norm/(grad_norm + 1.0e-8))` | max_norm=1.0, grad_norm=1.0 |
| `gradient_clip_value` | `Max(-clip_val, Min(clip_val, grad))` | clip_val=1.0 |
| `inverse_sqrt_lr` | `lr_init/sqrt(Max(epoch, warmup_steps))` | lr_init=0.01, warmup_steps=4000 |
| `momentum_step` | `-grad*lr - mu_momentum*v + w` | lr=0.01, mu_momentum=0.9, v=0 |
| `nesterov_step` | `-grad_lookahead*lr - mu_momentum*v + w` | lr=0.01, mu_momentum=0.9, v=0, grad_lookahead=0 |
| `one_cycle_approx_lr` | `lr_max*(cos(pi*epoch/T_max) + 1)/2` | lr_max=0.01, T_max=100 |
| `polynomial_decay_lr` | `lr_end + (-lr_end + lr_init)*(-epoch/max_epochs + 1)**power_lr` | lr_init=0.01, lr_end=1e-06, max_epochs=100, power_lr=1 |
| `quadratic_bowl` | `a_qb*(x - x_opt)**2 + b_qb*(y - y_opt)**2` | a_qb=1, b_qb=1, x_opt=0, y_opt=0 |
| `rastrigin_2d` | `x**2 + y**2 - 10*cos(2*pi*x) - 10*cos(2*pi*y) + 20` | None |
| `rmsprop_step` | `-grad*lr/sqrt(v_rms + 1.0e-8) + w` | lr=0.001, v_rms=0, rho=0.9 |
| `rosenbrock` | `b_rb*(-x**2 + y)**2 + (a_rb - x)**2` | a_rb=1, b_rb=100 |
| `sgd_step` | `-grad*lr + w` | lr=0.01 |
| `step_decay_lr` | `gamma_decay**floor(epoch/step_size)*lr_init` | lr_init=0.01, gamma_decay=0.1, step_size=30 |
| `warmup_linear_lr` | `lr_target*Min(1, epoch/warmup_steps)` | lr_target=0.001, warmup_steps=1000 |

## Metrics (27)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `accuracy_elem` | `Piecewise((1, Eq(y_pred, y_true)), (0, True))` | None |
| `adjusted_r_squared` | `-(1 - r2)*(n_samples - 1)/(-n_features + n_samples - 1) + 1` | r2=0.9, n_samples=100, n_features=5 |
| `aic` | `2*k - 2*log_likelihood` | k=1, log_likelihood=0 |
| `balanced_accuracy` | `sensitivity/2 + specificity/2` | sensitivity=1, specificity=1 |
| `bic` | `k*log(n_obs) - 2*log_likelihood` | k=1, n_obs=100, log_likelihood=0 |
| `brier_score_elem` | `(p - y)**2` | None |
| `cosine_distance_metric` | `-dot_ab/(norm_a*norm_b + 1.0e-8) + 1` | dot_ab=1, norm_a=1, norm_b=1 |
| `dcg_elem` | `rel*log(2)/log(pos + 1)` | rel=1, pos=1 |
| `dice_coefficient` | `2*n_intersect/(size_a + size_b + 1.0e-8)` | n_intersect=1, size_a=1, size_b=1 |
| `explained_variance_ratio` | `lambda_i/(total_variance + 1.0e-8)` | lambda_i=1, total_variance=10 |
| `f1_score_formula` | `2*prec*rec/(prec + rec + 1.0e-8)` | prec=1, rec=1 |
| `fbeta_score_formula` | `prec*rec*(beta_f**2 + 1)/(beta_f**2*prec + rec + 1.0e-8)` | prec=1, rec=1, beta_f=1 |
| `huber_loss_elem` | `Piecewise(((y_pred - y_true)**2/2, delta >= Abs(y_pred - y_true)), (delta*(-delta/2 + Abs(y_pred - y_true)), True))` | delta=1.0 |
| `jaccard_index` | `n_intersect/(n_union + 1.0e-8)` | n_intersect=1, n_union=1 |
| `log_loss_elem` | `-y*log(p + 1.0e-15) - (1 - y)*log(1.0 - p)` | None |
| `mae_elem` | `Abs(y_pred - y_true)` | None |
| `mape_elem` | `Abs((y_pred - y_true)/(y_true + 1.0e-8))` | None |
| `matthews_corr` | `(-fn*fp + tn*tp)/(sqrt((fn + tn)*(fn + tp)*(fp + tn)*(fp + tp)) + 1.0e-8)` | tp=1, tn=1, fp=0, fn=0 |
| `mse_elem` | `(y_pred - y_true)**2` | None |
| `ndcg_formula` | `dcg/(idcg + 1.0e-8)` | dcg=1, idcg=1 |
| `precision_formula` | `tp/(fp + tp + 1.0e-8)` | tp=1, fp=0 |
| `r_squared` | `-ss_res/(ss_tot + 1.0e-8) + 1` | ss_res=0, ss_tot=1 |
| `recall_formula` | `tp/(fn + tp + 1.0e-8)` | tp=1, fn=0 |
| `reciprocal_rank` | `1/rank` | rank=1 |
| `rmse_formula` | `sqrt(mse_val)` | mse_val=1 |
| `smape_elem` | `Abs(y_pred - y_true)/(Abs(y_pred)/2 + Abs(y_true)/2 + 1.0e-8)` | None |
| `specificity_formula` | `tn/(fp + tn + 1.0e-8)` | tn=1, fp=0 |

## Transforms (33)

| Name | Symbolic expression | Default parameters |
|---|---|---|
| `box_cox_approx` | `(x**lambda_bc - 1)/lambda_bc` | lambda_bc=0.5 |
| `clip_transform` | `Max(low, Min(high, x))` | low=0, high=1 |
| `difference_transform` | `x - x_prev` | x_prev=0 |
| `exponential_smoothing` | `alpha_es*x + s_prev*(1 - alpha_es)` | alpha_es=0.3, s_prev=0 |
| `fourier_feature_cos` | `cos(2*pi*freq*x)` | freq=1 |
| `fourier_feature_sin` | `sin(2*pi*freq*x)` | freq=1 |
| `gaussian_basis` | `exp(-(-center + x)**2/(2*width**2))` | center=0, width=1 |
| `he_normal_std` | `sqrt(2)*sqrt(1/fan_in)` | fan_in=256 |
| `he_uniform_bound` | `sqrt(6)*sqrt(1/fan_in)` | fan_in=256 |
| `l2_normalize` | `x/(norm_l2 + 1.0e-8)` | norm_l2=1 |
| `lecun_normal_std` | `sqrt(1/fan_in)` | fan_in=256 |
| `log10_transform` | `log(x + 1)/log(10)` | None |
| `log_return` | `log(x/(x_prev + 1.0e-8))` | x_prev=1 |
| `log_transform` | `log(x + 1)` | None |
| `max_abs_scale` | `x/(max_abs + 1.0e-8)` | max_abs=1 |
| `min_max_normalize` | `(x - x_min)/(x_max - x_min + 1.0e-8)` | x_min=0, x_max=1 |
| `min_max_to_range` | `a_range + (-a_range + b_range)*(x - x_min)/(x_max - x_min + 1.0e-8)` | x_min=0, x_max=1, a_range=-1, b_range=1 |
| `moving_average_weight` | `1/window_size` | window_size=5 |
| `positional_enc_cos` | `cos(pos/10000**(2*i/d_model))` | pos=1, i=0, d_model=512 |
| `positional_enc_sin` | `sin(pos/10000**(2*i/d_model))` | pos=1, i=0, d_model=512 |
| `power_transform` | `x**p_exp` | p_exp=2.0 |
| `robust_scale` | `(-median_val + x)/(iqr + 1.0e-8)` | median_val=0, iqr=1 |
| `sigmoid_transform` | `1/(1 + exp(-k_sig*(x - x0_sig)))` | k_sig=1, x0_sig=0 |
| `softplus_transform` | `log(exp(x) + 1)` | None |
| `softsign_transform` | `x/(Abs(x) + 1)` | None |
| `sqrt_transform` | `sqrt(x)` | None |
| `tanh_transform` | `tanh(x)` | None |
| `thermometer_bit` | `Piecewise((1, threshold <= x), (0, True))` | threshold=0.5 |
| `winsorize` | `Max(lower_pct, Min(upper_pct, x))` | lower_pct=-3, upper_pct=3 |
| `xavier_normal_std` | `sqrt(2)*sqrt(1/(fan_in + fan_out))` | fan_in=256, fan_out=256 |
| `xavier_uniform_bound` | `sqrt(6)*sqrt(1/(fan_in + fan_out))` | fan_in=256, fan_out=256 |
| `yeo_johnson_positive` | `((x + 1)**lambda_yj - 1)/lambda_yj` | lambda_yj=0.5 |
| `z_score_normalize` | `(-mu + x)/(sigma + 1.0e-8)` | mu=0, sigma=1 |

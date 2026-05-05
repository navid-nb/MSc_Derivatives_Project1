# Derivatives — Option Pricing & Implied Volatility (AAPL, COVID-19 Period)

Applied option pricing and implied volatility analysis on Apple (AAPL) equity options using OptionMetrics data from two contrasting market regimes: January 17, 2020 (pre-COVID) and March 20, 2020 (peak COVID panic). The assignment covers European and American option pricing, IV inversion, volatility smile dynamics, early exercise premiums, and implied stock borrow fees.

---

## Methods & Models

**1. Term Structure of Risk-Free Rates in BMS**
Analyzed how the yield curve and COVID-driven rate cuts affect the risk-free rate used in the Black-Merton-Scholes (BMS) model, and justified using maturity-matched rates within the framework.

**2. Discrete Dividend Adjustment (BMS with Dividends)**
Applied the ex-dividend stock price adjustment (`stock_exdiv`) to account for discrete cash dividends in the BMS formula:
$$F(t,T) = e^{r(T-t)}\bigl(S(t) - \textstyle\sum_n d(t_n)e^{-r(t_n-t)}\bigr)$$
Examined Apple's dividend schedule across both dates and confirmed no change in dividend policy.

**3. BMS Implied Volatility Inversion**
Implemented IV extraction by numerically inverting the dividend-adjusted BMS formula for each OTM option (puts: M ≤ 1, calls: M > 1). Used both Brent's method (`brentq`) and Golden Section Search for root-finding. Validated results against provider-supplied implied volatilities.

**4. Volatility Smile & Smirk Analysis**
Constructed volatility smiles across moneyness and maturity for both dates. Identified:
- Steeper OTM put smiles driven by crash-insurance demand
- Volatility smirk in March 2020 OTM calls reflecting asymmetric tail-risk pricing
- Term structure of implied volatility (near-term options show steeper smiles)

Introduced standardized moneyness $\ln(K/S)/(\sigma_{\text{ATM}}\sqrt{\tau})$ to normalize smiles across maturities and volatility regimes for meaningful cross-date comparison.

**5. CRR Binomial Tree — American Option Pricing**
Built a vectorized Cox-Ross-Rubinstein (CRR) binomial tree for American options with continuous dividend yield. Used N = 5 × DTM steps and Brent's method to extract CRR-implied volatilities. Ran 1,658 options in parallel using `joblib`. Compared BMS vs. CRR implied volatilities against the provider benchmark (MAE: BMS 0.0021 vs. CRR 0.0035), finding BMS more accurate due to discrete-vs-continuous dividend approximation error in CRR.

**6. Early Exercise Premium**
Estimated the early exercise premium for American options by pricing equivalent European options (using BMS with provider IV and convenience yield) and differencing from American market prices. Applied the economic bound that European price ≤ American price. Showed that deep ITM puts carry the largest early exercise value, while calls carry near-zero premiums on a dividend-paying stock.

**7. Implied Stock Borrow Fee (Muravyev, Pearson & Pollet)**
Computed the implied annualized stock borrow fee from ATM put-call IV spreads using the approximation from Muravyev, Pearson and Pollet (2022/2025):
$$h_t^{\mathbb{Q}} \approx -(\sigma_c - \sigma_p)/\sqrt{2\pi(T-t)}$$
Results: near-zero borrow fee in January 2020 (balanced put-call parity), widening to −2.2% in March 2020, reflecting elevated put demand and downside protection costs during the COVID crash.

---

## Key Files

| File | Description |
|------|-------------|
| [`202510-assignment1-en.ipynb`](202510-assignment1-en.ipynb) | Main notebook — full pipeline with all questions, code, and written analysis |
| [`202510-assignment1-en.html`](202510-assignment1-en.html) | Rendered HTML report with all outputs and figures (recommended for quick review) |
| [`helpers/BMS.py`](helpers/BMS.py) | BMS pricing with discrete dividends; IV inversion via Brent and Golden Section |
| [`helpers/CRR_tree.py`](helpers/CRR_tree.py) | Vectorized CRR binomial tree for American options; parallel IV extraction |
| [`helpers/plotting.py`](helpers/plotting.py) | All figure generation functions (smile plots, early exercise, borrow fee) |

---

## Tools & Libraries

Python · NumPy · pandas · SciPy (`brentq`, `golden`) · joblib (parallel processing) · OptionMetrics (data source) · Matplotlib

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open the main notebook:

```bash
jupyter notebook 202510-assignment1-en.ipynb
```

The dataset is pre-loaded from `202510-assignment1.pkl`. To regenerate from OptionMetrics, set `GENERATE = True` in the first code cell (requires OptionMetrics access).

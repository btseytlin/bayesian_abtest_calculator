import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import streamlit as st
from matplotlib import pyplot as plt
from plotly import express as px
from scipy.stats import beta as beta_distr

st.title("Bayesian A/B test for conversion rate")

st.subheader("Prior")


def visualize_prior(alpha, beta):
    p_h_values_range = np.arange(1e-10, 1, 0.01)

    prior_distr = beta_distr(alpha, beta)

    plot_df = pd.DataFrame(
        {
            "x": p_h_values_range,
            "p(p_h=x)": prior_distr.pdf(p_h_values_range),
        }
    )
    fig = px.line(
        plot_df,
        x="x",
        y="p(p_h=x)",
        title=f"Beta prior pdf with parameters a={alpha}, b={beta}",
    )
    fig.update_yaxes(range=[0, 1.2])
    return fig


alpha = st.number_input("Beta prior $\\alpha$", min_value=1)
beta = st.number_input("Beta prior $\\beta$", min_value=10)
st.plotly_chart(visualize_prior(alpha, beta))

st.subheader("Observed data")

n_A = st.number_input("Number of visitors in group A", min_value=1, value=100)
conv_A = st.number_input("Observed conversions in group A", min_value=1, value=10)
n_B = st.number_input("Number of visitors in group B", min_value=1, value=100)
conv_B = st.number_input("Observed conversions in group B", min_value=1, value=10)

st.subheader("Posterior")


clicked = st.button("Run simulation and plot posterior")

if clicked:

    def get_posterior(alpha, beta, n_A, conv_A, n_B, conv_B):
        with pm.Model() as model:
            p_A = pm.Beta("p_A", alpha, beta)
            p_B = pm.Beta("p_B", alpha, beta)

            p_A_obs = pm.Binomial("p_A_obs", n_A, p_A, observed=conv_A)
            p_B_obs = pm.Binomial("p_B_obs", n_B, p_B, observed=conv_B)

            trace = pm.sample(5000, chains=4, cores=2)

        return trace

    with st.spinner(text="In progress..."):
        posterior_trace = get_posterior(alpha, beta, n_A, conv_A, n_B, conv_B)
    samples_A = posterior_trace.posterior["p_A"].values
    samples_B = posterior_trace.posterior["p_B"].values

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    az.plot_posterior(posterior_trace, ax=[ax1, ax2])
    st.pyplot(fig)

    st.text(f"Probability that B is better: {(samples_B > samples_A).mean():.1%}.")

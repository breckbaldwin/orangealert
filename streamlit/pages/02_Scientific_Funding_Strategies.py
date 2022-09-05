import streamlit as st
from numpy.random import default_rng
import random
import pandas as pd
import plotnine

st.write("")

def run_sim(num_projects, num_funding_rounds, num_funding_slots, random_above_threshold, top_N):
    rng = default_rng()
    projects_top_n = [{}] * num_projects
    projects_random_n = [{}] * num_projects
    names = "abcdefghijklmnopqrstuvwxyz"
    for i in range(num_projects):
        projects_top_n[i] = {'id': f"Project {names[i]}", 'reputation': .1, 'merit': .1, 'resources': 0}
        if i%10 == 0:
            projects_top_n[i]['merit'] += 0
        projects_random_n[i] = {'id': names[i], 'reputation': .1, 'merit': .1, 'resources': 0}
        if i%10 == 0:
            projects_random_n[i]['merit'] += 0
    threshold = 0.2
    sd_merit = 0.1
    x = []
    y = []
    y2 = []
    pkg = []
    for rfp in range(1, num_funding_rounds + 1):
        for i in range(0, num_projects):
            projects_top_n[i]['score'] = rng.normal(projects_top_n[i]['merit'], sd_merit, 1)[0] +\
                                         projects_top_n[i]['reputation']
            projects_random_n[i]['score'] = rng.normal(projects_random_n[i]['merit'], sd_merit, 1)[0] +\
                                            projects_random_n[i]['reputation']
        sorted_candidates = sorted(projects_top_n, 
                    key=lambda s:['score'], reverse=True)    
        for winner in sorted_candidates[0:num_funding_slots]:
            winner['reputation'] += .1
            winner['resources'] += 1
        random_candidates = []
        threshold_drop = 0.0
        while len(random_candidates) < num_funding_slots:
            random_candidates = [s for s in projects_random_n if s['score'] > threshold - threshold_drop]
            if len(random_candidates) < num_funding_slots:
                threshold_drop += .01
        if threshold_drop > 0:
            st.info(f"Standards have been lowered! Threshold lowered by {threshold_drop:.2f} for iteration {rfp}")
        for winner in random.sample(random_candidates, num_funding_slots):
            winner['reputation'] += .1
            winner['resources'] += 1

        jitter = .1
        for i in range(0, num_projects):
            #x.append(rng.normal(rfp, jitter, 1)[0])
            x.append(rfp)
            y.append(rng.normal(projects_top_n[i]['resources'], jitter + .1, 1)[0])
            y2.append(rng.normal(projects_random_n[i]['resources'], jitter, 1)[0])
            pkg.append(projects_top_n[i]['id'])

    package_label = "Top N ___\nAbove Threshold N ..."

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df['resources'] = y
    df2['resources'] = y2
    df['rfp_count'] = x
    df2['rfp_count'] = x
    df['selection method'] = ['Top N totally ordered by score'] * len(x)
    df2['selection method'] = ['Random N score above threshold'] * len(x)
    df[package_label] = pkg
    df2[package_label] = pkg

    plot = (plotnine.ggplot(mapping=plotnine.aes(x='rfp_count',          y='resources', group = package_label)))
    if top_N:
        plot = plot + plotnine.geom_line(data=df, mapping=plotnine.aes(color=package_label), size=.7) 
    if random_above_threshold:    
        plot = plot + plotnine.geom_line(data=df2, mapping=plotnine.aes(color=package_label), size=.7, linetype='dotted')
    #plot = plot + \
    #plotnine.ggtitle(f"{num_funding_rounds} funding cycles, {num_projects} projects with {num_funding_slots} awards") +\
    #plotnine.labs(x="Iterations of funding cycle", y="Accumulated resources over time") + \ 
    plot = plot + plotnine.theme_xkcd()
    st.pyplot(plotnine.ggplot.draw(plot))


st.title("Simulating impact of peer review on supporting meritorious science")

exp = st.expander("Description")
exp.markdown("""

Contact: Breck Baldwin, breckbaldwin@gmail.com

Position paper at: https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html

All projects started with the same merit value = .1, and reputation value = 0. The score for a round of funding is a random draw from a normal distribution centered on merit with standard deviation .1 then added to the reputation value. Threshold for funding is .2 so candidates have to get a bit lucky to draw a high value for the score initially to clear the threshold and then get lucky by being drawn from the set of candidates above threshold. If a candidate is funded then their reputation increases .1 and the resource count increases by 1. Resources/reputation can only go up, merit stays the same. """)

(col1, col2) = st.columns(2)
num_projects = col1.slider("Number of projects?", min_value=1, max_value=20, step=1, value=10)
num_funding_rounds = col1.slider("How many rounds funding?", min_value=1, max_value=20, step=1, value=2)
num_funding_slots = col1.slider("Number of funding slots?", min_value=1, max_value=20, step=1, value=1)
random_above_threshold = col2.checkbox("Show Results for Random above Threshold", value=True)
top_N = col2.checkbox("Show Results for Top N Scored Proposals", value=True)


run_sim(num_projects, num_funding_rounds, num_funding_slots, random_above_threshold, top_N)
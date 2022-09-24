from kiwisolver import BadRequiredStrength
import streamlit as st
st.set_page_config(layout="wide")
from numpy.random import default_rng
import random
import pandas as pd
import plotnine as p9
from mizani.formatters import percent_format
import copy

st.title("How Algorithims Influence Research Diversity")
st.markdown("""**Breck Baldwin**, breckbaldwin@gmail.com
September, 2022""")

exp = st.expander("Description")
exp.markdown("""


Based on paper at: https://breckbaldwin.github.io/S3rd/presentations/DOE2021/FundingStrategiesForSciSoftware.html


**TL;DR** Ranking research proposals by quality and choosign Top N leads to concentration of resources to N organizations. Random N selection of proposals that are above a quality threshold distributes funding opportunties more broadly. 

- All projects started with the same merit and reputation values. 
- Score = The score for a round of funding is a random draw from a normal distribution centered on merit with standard deviation .1 then added to the reputation value.
- Top N: Award top N scored proposals for the round.
- Random N: Award random N selected from Score > funding threshold.
    + The default threshold for funding is .2, so candidates have to get a bit lucky to draw a high value for the score initially to clear the threshold and then get lucky by being drawn from the set of candidates above threshold.
- If a candidate is funded then their reputation increases .1 and the resource count increases by 1. Resources/reputation can only go up, merit stays the same. """)

RNG = default_rng()
NAMES = "abcdefghijklmnopqrstuvwxyz".upper()




def init(proj_start_values):
    top_n = [None] * len(proj_start_values)
    random_n = [None] * len(proj_start_values)
    hybrid = [None] * len(proj_start_values)
    proj_jitter = 0.035
    algo_jitter = 0.05
    for i in range(len(proj_start_values)):
        top_n[i] = {'id': f"Proj {NAMES[i]}", 
                    'algo': 'Top N',
                    'reputation': 0.0, 
                    'skill': proj_start_values[i], 
                    'total funds': -.5 + i * proj_jitter,
                    'cumulative benefit': 0}
        random_n[i] = {'id': f"Proj {NAMES[i]}", 
                        'algo': 'Random N',
                        'reputation': 0.0, 
                        'skill': proj_start_values[i], 
                        'total funds': algo_jitter + i * proj_jitter,
                        'cumulative benefit': 0}
        hybrid[i] = {'id': f"Proj {NAMES[i]}", 
                        'algo': 'Hybrid',
                        'reputation': 0.0, 
                        'skill': proj_start_values[i], 
                        'total funds': - (algo_jitter + i * proj_jitter),
                        'cumulative benefit': 0}
    return (top_n, random_n, hybrid)

def compute_benefit(winner):
    if winner['cumulative benefit'] == 0:
        winner['cumulative benefit'] = 1
    else:
        winner['cumulative benefit'] += winner['cumulative benefit'] * .75

def select_random_n(num_funding_slots, rand_n, funding_threshold):
    random_candidates = []
    threshold_drop = 0.0
    while len(random_candidates) < num_funding_slots:
        random_candidates = [s for s in rand_n if s['score'] > 
                                funding_threshold - threshold_drop]
        if len(random_candidates) < num_funding_slots:
            threshold_drop += .5
    if threshold_drop > 0:
        st.info(f"Standards have been lowered! Threshold lowered by {threshold_drop:.2f} for iteration")
    return random.sample(random_candidates, num_funding_slots)

def run3(top_n, rand_n, hybrid, num_funding_rounds, num_projects, budget,
            reputation_increase_from_funding, funding_threshold):
    rounds_funding = []
    for funding_round in range(num_funding_rounds + 1):
        for proj_round in top_n + rand_n + hybrid:
            result = copy.deepcopy(proj_round)
            result['round'] = funding_round
            rounds_funding.append(result)
        if funding_round > num_funding_rounds:
            break
        for i in range(0, num_projects):
            proposal_quality = RNG.normal(0, 1.0) #draw from bell curve
            top_n[i]['score'] = (proposal_quality + 
                                top_n[i]['skill'] + 
                                top_n[i]['reputation'])
            rand_n[i]['score'] = (proposal_quality +
                                  rand_n[i]['skill'] +
                                  rand_n[i]['reputation'])
            hybrid[i]['score'] = (proposal_quality +
                                hybrid[i]['skill'] +
                                hybrid[i]['reputation'])    
    #top_n
        sorted_candidates = sorted(top_n, 
                                key=lambda s:['score'], 
                                reverse=True)    
        for winner in sorted_candidates[0:budget]:
            winner['reputation'] +=  reputation_increase_from_funding
            winner['total funds'] += 1
            #winner['cumulative benefit'] = compute_benefit(winner)
    #random_n
        winners = select_random_n(budget, rand_n, funding_threshold)
        for winner in winners:
            winner['reputation'] += reputation_increase_from_funding
            winner['total funds'] += 1
    #hybrid
        sorted_candidates = sorted(hybrid, 
                                key=lambda s:['score'], 
                                reverse=True)
        top_n_slots = budget // 2
        top_n_winners = sorted_candidates[0:top_n_slots]
        random_candidates = [candidate for candidate in hybrid if 
                                candidate not in top_n_winners]
        random_n_winners = select_random_n(budget - top_n_slots,
                                            random_candidates,
                                            funding_threshold)
        for winner in top_n_winners + random_n_winners:
            winner['reputation'] += reputation_increase_from_funding
            winner['total funds'] += 1
    return rounds_funding

def render3(df, show_top_n, show_random_n, show_hybrid):
    if not (show_top_n or show_random_n or show_hybrid):
        st.write("select a dataset to view")
        return
    plot = (p9.ggplot(mapping=p9.aes(x='round', y='total funds', group = 'id')))
    if show_top_n:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Top N'], 
                                    mapping=p9.aes(color='id'), size=.7) 
    if show_random_n:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Random N'],   
                                    mapping=p9.aes(color='id'), size=.7, 
                                    linetype='dotted')
    if show_hybrid:
        plot = plot + p9.geom_line(data=df[df['algo'] == 'Hybrid'], 
                                    mapping=p9.aes(color='id'), size=.7, 
                                    linetype='dashdot')
    #plot = plot + p9.theme_xkcd()
    st.pyplot(p9.ggplot.draw(plot))


def run():
    (col1, col2) = st.columns(2)
    #Column 1
    num_projects = 10
    #NUM_PROJECTS = col1.slider("Number of projects?", min_value=2, max_value=20, step=2, value=10)
    num_funding_rounds = 4
    num_funding_rounds = col1.slider("How many funding cycles?", min_value=1, max_value=10, step=1, value=num_funding_rounds)
    budget = 3
    BadRequiredStrength = col1.slider("""Budget in millions per cycle--each
award is $1 million?""", 
    min_value=1, max_value=20, step=1, value=budget)
    num_simulations = 1
    num_simulations = col1.slider("Number of simulations (1 shows graph)", 
                                    min_value=1, max_value=100, step=10,
                                    value=num_simulations)
    hybrid_percent_top_n = .5
    hybrid_percent_top_n = col1.slider("Hybrid model percentage Top N",
                                        min_value=0, max_value=100, step=10)

    #Column 2
    col2.selectbox("Preconfigured Experiments", ['Balanced', 'Foor'])
    funding_threshold = 2.0 
    # FUNDING_THRESHOLD = col2.radio("Proposal score threshold for Random N selection", 
    #                                 ['0.0 No grade threshold', 
    #                                 '1.0 D grade or better', 
    #                                 '2.0 C grade or better', 
    #                                 '3.0 B grade or better', 
    #                                 '4.0 A grade'])
    reputation_bump_for_funding = col2.radio("Reputation increase if funded", 
                                            ['0.0 No reputation increase',
                                            '0.5 Half grade increse',
                                            '1.0 One grade increase'])
    reputation_increase_from_funding = 0.5

    impact_curve = col2.radio("Research productivity growth is:",
                                ['Funding increases productivty uniformly',
                                'Early rouds of funding create higher productivity than later rounds funding',
                                'Later rounds of funding have productivity than early rounds'])

   
    skills_display = ['0.0 F', '1.0 D', '2.0 C', '3.0 D', '4.0 A']
    skills = [0.0, 1.0, 2.0, 3.0, 4.0]
    skills.reverse()

    st.write("Initial Skills Assignment")
    proj_skill_values = [1.0] * num_projects
    cols = st.columns(num_projects)
    with st.form("Custom Initial Skills"):
        for i in range(num_projects):
            proj_skill_values[i] = \
                cols[i].radio(f"Proj {NAMES[i]}", 
                                skills, 
                                index=skills.index(proj_skill_values[i]),
                                key=i, 
                                horizontal=False)
    (top_n, random_n, hybrid) = init(proj_skill_values)

    top_n_df = pd.DataFrame(top_n)
    rand_n_df = pd.DataFrame(random_n)
    hybrid_df = pd.DataFrame(hybrid)

    exp = st.expander("Initialization Details--all projects should be the same across algorithms")
    (col1, col2, col3) = exp.columns(3)
    col1.write("Top N initial values")
    col1.dataframe(top_n_df)
    col2.write("Random N initial values")
    col2.dataframe(rand_n_df)
    col3.write("Hybrid initial values")
    col3.dataframe(hybrid)

    data_df = pd.DataFrame(run3(top_n, random_n, hybrid, num_funding_rounds,
                                num_projects, budget,
                                reputation_increase_from_funding,
                                funding_threshold))
    st.dataframe(data_df)
    show_random_n = st.checkbox("Show Random N", value=False, key="rand_n")
    show_hybrid = st.checkbox("Show Hybrid", value=False, key="hybr")
    show_top_n = st.checkbox("Show Top N", value=True, key="top_n")

    render3(data_df, show_top_n, show_random_n, show_hybrid)


run()

# def run(projects_top_n, projects_random_n):
#     threshold = 0.2
#     sd_merit = 0.1
#     x = []
#     y = []
#     y2 = []
#     pkg = []
#     for rfp in range(1, num_funding_rounds + 1):
#         for i in range(0, num_projects):
#             projects_top_n[i]['score'] = rng.normal(projects_top_n[i]['merit'], sd_merit, 1)[0] +\
#                                             projects_top_n[i]['reputation']
#             projects_random_n[i]['score'] = rng.normal(projects_random_n[i]['merit'], sd_merit, 1)[0] +\
#                                             projects_random_n[i]['reputation']
#         sorted_candidates = sorted(projects_top_n, 
#                     key=lambda s:['score'], reverse=True)    
#         for winner in sorted_candidates[0:num_funding_slots]:
#             winner['reputation'] += .1
#             winner['resources'] += 1

#         random_candidates = []
#         threshold_drop = 0.0
#         while len(random_candidates) < num_funding_slots:
#             random_candidates = [s for s in projects_random_n if s['score'] > threshold - threshold_drop]
#             if len(random_candidates) < num_funding_slots:
#                 threshold_drop += .01
#         if threshold_drop > 0:
#             st.info(f"Standards have been lowered! Threshold lowered by {threshold_drop:.2f} for iteration {rfp}")
#         for winner in random.sample(random_candidates, num_funding_slots):
#             winner['reputation'] += .1
#             winner['resources'] += 1
#         jitter = .1
#         for i in range(0, num_projects):
#             #x.append(rng.normal(rfp, jitter, 1)[0])
#             x.append(rfp)
#             y.append(rng.normal(projects_top_n[i]['resources'], jitter + .1, 1)[0])
#             y2.append(rng.normal(projects_random_n[i]['resources'], jitter, 1)[0])
#             pkg.append(projects_top_n[i]['id'])
#     return(x, y, y2, pkg)




    
# def render(x, y, y2, pkg):
#     package_label = "Top N ___\nAbove Threshold N ..."
#     df = pd.DataFrame()
#     df2 = pd.DataFrame()
#     df['resources'] = y
#     df2['resources'] = y2
#     df['rfp_count'] = x
#     df2['rfp_count'] = x
#     df['selection method'] = ['Top N totally ordered by score'] * len(x)
#     df2['selection method'] = ['Random N score above threshold'] * len(x)
#     df[package_label] = pkg
#     df2[package_label] = pkg

#     plot = (plotnine.ggplot(mapping=plotnine.aes(x='rfp_count',          y='resources', group = package_label)))
#     if top_N:
#         plot = plot + plotnine.geom_line(data=df, mapping=plotnine.aes(color=package_label), size=.7) 
#     if random_above_threshold:    
#         plot = plot + plotnine.geom_line(data=df2, mapping=plotnine.aes(color=package_label), size=.7, linetype='dotted')
#     #plot = plot + \
#     #plotnine.ggtitle(f"{num_funding_rounds} funding cycles, {num_projects} projects with {num_funding_slots} awards") +\
#     #plotnine.labs(x="Iterations of funding cycle", y="Accumulated resources over time") + \ 
#     plot = plot + plotnine.theme_xkcd()
#     st.pyplot(plotnine.ggplot.draw(plot))

# if st.checkbox("Run"):
#     top_n_counts = [0] * (num_funding_rounds + 1) 
#     random_n_counts = [0] * (num_funding_rounds + 1)
#     top_n_resources = []
#     rand_n_resources = []
#     for i in range(num_simulations):
#         (projects_top_n, projects_random_n) = init()
#         (x, y, y2, pkg) = run(projects_top_n, projects_random_n)
#         if num_simulations == 1:
#             render(x, y, y2, pkg)
#          #percent of projects with at resources >=N
#         top_n_df = pd.DataFrame(projects_top_n)
#         #st.dataframe(top_n_df)
#         random_n_df = pd.DataFrame(projects_random_n)
#         #st.dataframe(random_n_df)
#         for j in range(num_funding_rounds + 1):
#             top_n_counts[j] += len(top_n_df[top_n_df['resources'] == j])
#             random_n_counts[j] += len(random_n_df[random_n_df['resources'] == j])
#         top_n_resources.extend(list(top_n_df['resources']))
#         rand_n_resources.extend(list(random_n_df['resources']))
#     hist_df = \
#         pd.DataFrame({'Millions Awarded': top_n_resources + rand_n_resources,
#                     'Strategy': ['Top N'] * len(top_n_resources) + \
#                                 ['Random N'] * len(rand_n_resources)})
#     slider_vals = [0] * num_funding_rounds
#     for award_inc in range(1, num_funding_rounds):
#         slider_vals[award_inc] = st.slider(f"{award_inc}-{award_inc + 1}",
#                     min_value=-1.0, max_value=2.0, value=1.0, key=f"a_{award_inc}")
    
 
#     plot_top_n = (plotnine.ggplot(hist_df[hist_df['Strategy'] == 'Top N'], 
#                         plotnine.aes(x='Millions Awarded', 
#                                     #y=plotnine.after_stat('count'),
#                                      y=plotnine.after_stat('width*density'),
#                                      fill='Strategy'))
#             + plotnine.geom_histogram(binwidth=0.2)
#             + plotnine.scale_y_continuous(labels=percent_format(),
#                                           name="Percent Awarded Amount",
#                                           limits=[0,1])
#     )
#     plot_rand_n = (plotnine.ggplot(hist_df[hist_df['Strategy'] == 'Random N'], 
#                         plotnine.aes(x='Millions Awarded', 
#                                      y=plotnine.after_stat('width*density'),
#                                      #y=plotnine.after_stat('count'),
#                                      fill='Strategy'))
#             + plotnine.geom_histogram(binwidth=0.2)
#             + plotnine.scale_y_continuous(labels=percent_format(), 
#                                           name="Percent Awarded Amount",
#                                           limits=[0,1])
#     )
#     (col1, col2) = st.columns(2)
#     col1.pyplot(plotnine.ggplot.draw(plot_top_n))
#     col2.pyplot(plotnine.ggplot.draw(plot_rand_n))
#     col1.write(f"Top N: {top_n_counts} {sum(top_n_counts)}")
#     col2.write(f"Random N: {random_n_counts} {sum(random_n_counts)}")
    


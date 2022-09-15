import streamlit as st
#st.set_page_config(layout="wide")
import pandas as pd
import os
import sys
import copy
sys.path.append("pages/")

import util




st.title("How Algorithims Influence Research Diversity")
st.markdown("A simulation fueled exploration")
st.markdown("""**Breck Baldwin**, breckbaldwin@gmail.com
September, 2022""")


exp = st.expander("Introduction", expanded=False)
exp.markdown("""
## Welcome to the simulation

This is going to be a _little_ different from a standard blog post and an experiment in exposing you, gentle reader, (Y,GR) to a more dynamic way to explore ideas you may not be familliar with. 

## Where is the juice? 

To put it bluntly, this article talks about approaches to research grant funding which I assume you don't really have opinions about or particularly care about. The juice or what you get out of reading/interacting with this is:
- Lots of important decisions get made with the same algorithms I will be covering. Admissions, job applications and who gets picked for each side in a school yard kickball game.
- Y,GR will learn properties of the algorithms by running simulations with brutally short explanations. Experiential learners rejoyce! I'm your boy.
- No rule of three third point--got simulations to cover....

## Scoring candidates

Selection of 'winners' of limited resources often score candidates as a first step. This work focuses on what is done **after** the candidate scoring step has happened. But we still need a scoring step. The structure is as follows:

- We have 10 projects, A-J
- The projects have a 'skill' value between 0.0, an F, to 4.0, an A on US style grading scales. 
- The projects can have the same skill or different skills, you get to play around with it.
""")

exp.markdown("""### Simulate score as a function of skill

You'll be simulating wrecked lives as well as meteoric ascensions to greatness in no time, be patient. Who lets Y,GR play god? We do.

Steps are:

1. Assign skill values to the projects. The default is everyone is a '1.0'. If you choose to remain a communist you can leave them, otherwise you can mix them up a bit. 
2. There is a 'Bell Curve' button to recreate Y,GR's harsher grading environments. 
3. The skill is unavailable to mere human evaluators, but Y,GR are functioning as god here-- so you get access to the source code.
""")

num_projects = 10
skills = [0.0, 1.0, 2.0, 3.0, 4.0]
skills.reverse()
names = "abcdefghijklmnopqrstuvwxyz".upper()
names
st.write("Initial Skills Assignment")
proj_skill_values = [1.0] * num_projects

if st.checkbox("Show how an American does it! (Bell curve)", value=False):
    proj_skill_values = [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 2.0, 1.0]
cols = st.columns(num_projects)
for i in range(num_projects):
    proj_skill_values[i] = \
        cols[i].radio(f"Proj {names[i]}", 
                        skills, 
                        index=skills.index(proj_skill_values[i]),
                        key=i, 
                        horizontal=False)
y_offset = .03
(proj_data) = util.init(proj_skill_values, names)

#top_n_df = pd.DataFrame(top_n)

exp = st.expander("Stupid Human Simulation: Drawing a score from the bell curve", expanded=False)
exp.markdown("""
Now we have some place to start before we pitch off into the dreaded algorithims. Just a bit more to do before the **judging** begins.

As alluded to above, only god, knows the true skill behind a project. This could be people, resources, research area etc... But mere humans will be attempting to assign a score to each project's funding application. 

Starting to drift off? Swap out 'score' with 'college application', 'Mr. America profile', 'audition' or any of how much one's 'jib' the-cut-of is apprecatiated by another human. 

## Smart humans who figured out they are stupid

This score business is gonna be noisy. For example the smartest people at the world's smartest artificial intelligence conference (Neurips) can't agree on the quality of submitted research papers. Respect to the smart people who worked hard to figure out they were stupid--twice. To be fair the entire task is roughly impossible, but there are places of agreement--more on that later. 

Above you had the option to use the bell-curve to assign skill as god which in this simulation is the truth and a statement about how true skill varies among applicants. But now we are asking a mere human to **play** god by assigning a score to a research proposal that will be used to make a decision. If we had 100 reviewers we could assume the god assigned skill to be recovered most often of any score. The next most common scores would be plus/minus one grade (1.0), even less common would be those scores plus/minus 2.0 and so on. 

But just like whether Y,GR chose a bell curve for true skill, the bell curve, aka Normal or Gaussian distribution, is an assumption. But as an assumption it covers a broad range of sins and problems that I'll decriment my bullet-point budget to articulate:

- The key idea behind the bell curve as an error model for measurement is that errors tend to cancel each other out but only on average. So Dr. Faultenroy's utter ignorance of Bayesian statistics is canceled out by the fact that the title had a good pun. 
- Because averages don't always end up at 50\% for two choices you get skewed heads/tails lots of times.
- But most of the time you get 50/50, next most 49/51, 51/49, next most 48/52 and so on.
- In violation of the 3 bullet point rule: There are other error models, bell curve is just very common and robust. 

<img bell curve for coins>

That's enough for now, go take a statistics course if you want to know more and for the love of god make sure it is a Bayesian one.

So how do we manage our stupid human reviewer simulation? We assume they are going to be wrong a bunch but on average they will hit their numbers or be pretty close. We "virtually" throw a dart at the dark area of the above bell curve and if we hit the dark area we return the number on the x axis. If you think about it if we throw a million darts, the most common value will be the god decided skill. Pretty neat. But remember Y,GR, if you chose a bell curve for skills it has nothing to do with the measurements above, you could have been a unitarian. 

The approach to scoring is very simple. We draw, 'throw a dart', at the bell curve that is centered at 0, take the value, positive or negative, and add it to the skill. You can see the result in the below table.
""")

standard_deviation = .2
standard_deviation =\
    st.slider("How much do scores (evaluation) vary in +/- grade points",   
              min_value=0.0, max_value=1.0, step=0.1, value=standard_deviation)

util.add_score([proj_data], standard_deviation, 0)

exp = st.expander("Show drawn Scores", expanded=False)
proj_df = pd.DataFrame(proj_data)
exp.dataframe(proj_df.loc[:, proj_df.columns.isin(['id', 'skill', 'draw',
                                                 'score', 'round'])])
exp.markdown("""
Each round of funding will draw a score and add it to the skill + reputation scores for the project. The reputation is 0 now, but with successful funding it will grow which reflects the benefit of a project being funded for subsequent rounds of funding. Reputation is how the rich get richer in this simulation which may or may not be a good idea--and it is central to the algorithms that we are experimenting with below.
""")

exp = st.expander("The Top N Algorithm")
exp.markdown("""
## Algorithmic Meritocracy: Top N

The Top N algorithm will take the available budget, constrained to \$1 million awards, and parcel out budget starting at the top scoring project. If we have \$3 million in the budget, then the top 3 scoring projects get funding. If there are ties for the score then pick from the order that happens to be in the list. 

Since the default order in this implementation will always choose alphabetically 'Proj A' before 'Proj B' if they have the same score we see programmer lazyness introducing bias.

Repeating the algorithm:

1. Select Top N proposals by score where N = millions of dollars of budget.
2. For each awarded proposal, increment the total funding by $1 million 
3. For each awarded proposal Increment the reputation by .5 (half a grade point)

## Lets spend some money!

Below we have the controls for a Top N algorithm simulation. 


""")

#Column 1

#NUM_PROJECTS = col1.slider("Number of projects?", min_value=2, max_value=20, step=2, value=10)
budget = 3
budget = st.slider("""Budget in millions per cycle--each
award is $1 million?""", 
min_value=1, max_value=10, step=1, value=budget)
exp = st.expander("Show one round of funding")
top_n_data = copy.deepcopy(proj_data)
top_n_winners = util.select_top_n(top_n_data, budget)
funding_amount_in_millions = 1.0
reputation_increase_per_funding_round = 0.5

util.distribute_awards(top_n_winners, funding_amount_in_millions,
                         reputation_increase_per_funding_round)

proj_df = pd.DataFrame(top_n_data)
(col1, col2) = exp.columns(2)
col1.write("All Projects")
col1.dataframe(proj_df.loc[:, proj_df.columns.isin(['id', 'skill', 'draw',
                                                 'score', 'total funds'])])
top_n_winners_df = pd.DataFrame(top_n_winners)
col2.write("Winning Projects")
col2.dataframe(top_n_winners_df.loc[:, proj_df.columns.isin(['id', 'skill', 'draw',
                                                 'score', 'total funds'])])

exp = st.expander("Applying Top N algorithm for multiple rounds of funding")

exp.markdown("""
## Cumulative Effects of Many Rounds of Funding: Empire building

The Top N algorithm really shows its properties with repeated application. The key insight is the role of accumulated reputation which will give an advantage to projects that previously recieved funding. The analog in other domains would be generational wealth, personal wealth, and legacy applicants to colleges. 

The below slider controls the number of funding rounds which in turn creates a graph showing the accumulated funding for projects over time. 

""")

(col1, col2, col3) = st.columns(3)
num_funding_rounds = 4
num_funding_rounds =\
     col1.slider("How many funding cycles?", min_value=1, 
                max_value=10, step=1, value=num_funding_rounds)

reputation_increase_per_funding_round = .5
reputation_increase_per_funding_round =\
     col2.slider("How much increase in reputation per funding award", 
                min_value=0.0, 
                max_value=2.0, 
                step=.25,
                value=reputation_increase_per_funding_round)

minimum_threshold = 1.5
minimum_threshold = col3.slider("Minimum threshold for funding",
                                min_value=0.0,
                                max_value=3.0,
                                step=.5,
                                value=minimum_threshold)

(col1, col2, col3) = st.columns(3)
show_top_n = col1.checkbox("Top N", value=True)
show_rand_n = col2.checkbox("Random N", value=True)
show_hybrid = col3.checkbox("Hybrid", value=False)

proj_data_2 = util.init(proj_skill_values, names)

top_n = copy.deepcopy(proj_data_2)
for proj in top_n:
    proj['algo'] = 'Top N'

rand_n = copy.deepcopy(proj_data_2)
for proj in rand_n:
    proj['algo'] = 'Random N'

hybrid = copy.deepcopy(proj_data_2)
for proj in hybrid:
    proj['algo'] = 'Hybrid'

top_n = copy.deepcopy(proj_data_2)
results = []
results.extend(top_n)
results.extend(rand_n)
results.extend(hybrid)
for round_num in range(1, num_funding_rounds + 1):
    top_n = copy.deepcopy(top_n)
    rand_n = copy.deepcopy(rand_n)
    hybrid = copy.deepcopy(hybrid)
    util.add_score([top_n, rand_n, hybrid], standard_deviation, round_num)

    top_n_winners = util.select_top_n(top_n, budget)
    util.distribute_awards(top_n_winners, funding_amount_in_millions,
                           reputation_increase_per_funding_round)
    results.extend(top_n)

    random_winners = util.select_random_n(budget, rand_n, minimum_threshold)
    util.distribute_awards(random_winners, funding_amount_in_millions,
                            reputation_increase_per_funding_round)
    results.extend(rand_n)

df = pd.DataFrame(results)

y_dimension = 'total funds'
y_dimension = st.radio("Y dimension", options=['total funds', 'draw', 'reputation', 'score', 'skill'])

import plotnine as p9

def render3(df, column_to_show, show_top_n, show_random_n, show_hybrid):
    if not (show_top_n or show_random_n or show_hybrid):
        st.write("select a dataset to view")
        return
    offset_scale =  (max(df[column_to_show]) - min(df[column_to_show])) / 100
    offset_scale = max(offset_scale, .01)
    df['y'] = df[column_to_show] + df['y_offset'] * offset_scale

    plot = (p9.ggplot(mapping=p9.aes(x='round', y='y', group = 'id')))
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
    if (column_to_show == 'skill' 
        and max(df[column_to_show]) == min(df[column_to_show])):
        single_y = max(df[column_to_show])
        plot = plot + p9.scale_y_continuous(
            label=f"Y projects jittered by {offset_scale:.2f}",
            limits=[single_y - 0.5, single_y + .5])
    
    plot = plot + p9.labels.ylab(f"Y projects jittered by {offset_scale:.2f}")
    #plot = plot + p9.theme_xkcd()
    st.pyplot(p9.ggplot.draw(plot))


render3(df, y_dimension, show_top_n, show_rand_n, show_hybrid)


f = """
 which then have a selection algorithm: Top N Pick the highest score and then allocate the resource. If there are resources left then pick the next highest scoring winner and so on. 

- Score candidates by merit

"""

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



def page():
    (col1, col2) = st.columns(2)
    #Column 1
    num_projects = 10
    #NUM_PROJECTS = col1.slider("Number of projects?", min_value=2, max_value=20, step=2, value=10)
    num_funding_rounds = 4
    num_funding_rounds = col1.slider("how many funding cycles?", min_value=1, max_value=10, step=1, value=num_funding_rounds)
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
    names = "abcdefghijklmnopqrstuvwxyz".upper()
    st.write("Initial Skills Assignment")
    proj_skill_values = [1.0] * num_projects
    cols = st.columns(num_projects)
    with st.form("Custom Initial Skills"):
        for i in range(num_projects):
            proj_skill_values[i] = \
                cols[i].radio(f"Proj {names[i]}", 
                                skills, 
                                index=skills.index(proj_skill_values[i]),
                                key=i, 
                                horizontal=False)
    (top_n, random_n, hybrid) = util.init(proj_skill_values, names)

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

    data_df = pd.DataFrame(util.run3(top_n, random_n, hybrid, 
                                num_funding_rounds,
                                num_projects, budget,
                                reputation_increase_from_funding,
                                funding_threshold))
    st.dataframe(data_df)
    show_random_n = st.checkbox("Show Random N", value=False, key="rand_n")
    show_hybrid = st.checkbox("Show Hybrid", value=False, key="hybr")
    show_top_n = st.checkbox("Show Top N", value=True, key="top_n")

    util.render3(data_df, show_top_n, show_random_n, show_hybrid)


page()

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
    


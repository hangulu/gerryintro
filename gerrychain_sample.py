# -*- coding: utf-8 -*-
"""
This script gets you started with the GerryChain package.
"""

### https://gerrychain.readthedocs.io/en/latest/user/recom.html
### https://gerrychain.readthedocs.io/en/latest/user/quickstart.html

## Import needed functions from library
from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
import pandas as pd
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, proposals, updaters, constraints, accept)
from gerrychain.tree_proposals import recom
from functools import partial
import tqdm


## Parameters
pennDataPathPrefix = ""
shpFileSuffix = "PA_VTD.shp"
jsonFileSuffix = "PA_VTD.json"



#### Import Data

#### The Graph.from_file() classmethod creates a Graph of the precincts in our
#### shapefile. By default, this method copies all of the data columns from the
#### shapefile’s attribute table to the graph object as node attributes.
#### The contents of this particular shapefile’s attribute table are summarized
#### in the mggg-states/PA-shapefiles GitHub repo.

#### Depending on the size of the state, the process of generating an adjacency
#### graph can take a bit of time. To avoid having to repeat this process in
#### the future, we call graph.to_json() to save the graph in the NetworkX
#### json_graph format under the name "PA_VTD.json.

graph = Graph.from_file(pennDataPathPrefix+shpFileSuffix)

graph.to_json(pennDataPathPrefix+jsonFileSuffix)

## Simple Example

#### In order to run a Markov chain, we need an adjacency Graph of our VTD
#### geometries and Partition of our adjacency graph into districts.
#### This Partition will be the initial state of our Markov chain.

#### We configure an Election object representing the 2012 Senate
#### election, using the USS12D and USS12R vote total columns from our
#### shapefile. The first argument is a name for the election ("SEN12"), and
#### the second argument is a dictionary matching political parties to their
#### vote total columns in our shapefile. This will let us compute hypothetical
#### election results for each districting plan in the ensemble.

election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

#### Finally, we create a Partition of the graph. This will be the starting
#### point for our Markov chain.

initial_partition = Partition(
    graph,
    assignment="2011_PLA_1",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("TOT_POP", alias="population"),
        "SEN12": election
    }
)

#### With the "population" updater configured, we can see the total population
#### in each of our congressional districts. In an interactive Python session,
#### we can print out the populations like this:

for district, pop in initial_partition["population"].items():
    print("District {}: {}".format(district, pop))

#### Notice that partition["population"] is a dictionary mapping the ID of each
#### district to its total population (that’s why we can call the .items()
#### method on it). Most updaters output values in this dictionary format.

#### For more information on updaters, see the gerrychain.updaters documentation.

#### Running a chain
#### Now that we have our initial partition, we can configure and run a Markov
#### chain. Let’s configure a short Markov chain to make sure everything works
#### properly.

chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

#### The above code configures a Markov chain called chain, but does not run it
#### yet. We run the chain by iterating through all of the states using a for
#### loop. As an example, let’s iterate through this chain and print out the
#### sorted vector of Democratic vote percentages in each district for each
#### step in the chain.

for partition in chain:
    print(sorted(partition["SEN12"].percents("Dem")))

#### That’s all: you’ve run a Markov chain!

#### To analyze the Republican vote percentages for each districting plan in
#### our ensemble, we’ll want to actually collect the data, and not just print
#### it out. We can use a list comprehension to store these vote percentages,
#### and then convert it into a pandas DataFrame.

d_percents = [sorted(partition["SEN12"].percents("Dem")) for partition in chain]

data = pd.DataFrame(d_percents)

#### This code will collect data from a different ensemble than our for loop
#### above. Each time we iterate through the chain object, we run a fresh new
#### Markov chain (using the same configuration that we defined when
#### instantiating chain).

#### The pandas DataFrame object has many helpful methods for analyzing and
#### plotting data. For example, we can produce a boxplot of our ensemble’s
#### Democratic vote percentage vectors, with the initial 2011 districting plan
#### plotted in red, in just a few lines of code:
ax = data.boxplot()
data.iloc[0].plot(style="ro", ax=ax)

plt.show()

#### (Before you over-analyze this data, keep in mind that this is a toy
#### ensemble of just one thousand plans created by single flips.)

#### Recon Sample
#### This document shows how to run a chain using the ReCom proposal used in
#### MGGG’s 2018 Virginia House of Delegates report.

#### Our goal is to use ReCom to generate an ensemble of districting plans for
#### Pennsylvania, and then make a box plot comparing the Democratic vote
#### shares for plans in our ensemble to the 2011 districting plan that the
#### Pennsylvania Supreme Court found to be a Republican-favoring partisan
#### gerrymander.

#### Setting up the initial districting plan
#### We configure Election objects representing some of the election data from
#### our shapefile.

elections = [
    Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
    Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
    Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
]

#### Configuring our updaters
#### We want to set up updaters for everything we want to compute for each plan
#### in the ensemble.

# Population updater, for computing how close to equality the district
# populations are. "TOT_POP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOT_POP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

#### Instantiating the partition
#### We can now instantiate the initial state of our Markov chain, using the 2011 districting plan:

initial_partition = GeographicPartition(graph,
                                        assignment="2011_PLA_1",
                                        updaters=my_updaters)

#### GeographicPartition comes with built-in area and perimeter updaters. We do
#### not use them here, but they would allow us to compute compactness scores
#### like Polsby-Popper that depend on these measurements.


#### Setting up the Markov chain
#### Proposal
#### First we’ll set up the ReCom proposal. We need to fix some parameters
#### using functools.partial before we can use it as our proposal function.

# The ReCom proposal needs to know the ideal population for the districts so that
# we can improve speed by bailing early on unbalanced partitions.

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="TOT_POP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )

#### Constraints
#### To keep districts about as compact as the original plan, we bound the
#### number of cut edges at 2 times the number of cut edges in the initial
#### plan.

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

#### Configuring the Markov chain
chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

#### Running the chain
#### Now we’ll run the chain, putting the sorted Democratic vote percentages
#### directly into a pandas DataFrame for analysis and plotting. The DataFrame
#### will have a row for each state of the chain. The first column of the
#### DataFrame will hold the lowest Democratic vote share among the districts
#### in each partition in the chain, the second column will hold the
#### second-lowest Democratic vote shares, and so on.

# This will take about 10 minutes.


#data = pd.DataFrame(
#    sorted(partition["SEN12"].percents("Democratic"))
#    for partition in chain
#)

#### If you install the tqdm package, you can see a progress bar as the chain
#### runs by running this code instead:

data = pd.DataFrame(
   sorted(partition["SEN12"].percents("Democratic"))
   for partition in chain.with_progress_bar()
)


#### Create a plot
#### Now we’ll create a box plot similar to those appearing the Virginia report.

fig, ax = plt.subplots(figsize=(8, 6))

# Draw 50% line
ax.axhline(0.5, color="#cccccc")

# Draw boxplot
data.boxplot(ax=ax, positions=range(len(data.columns)))

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
data.iloc[0].plot(style="ro", ax=ax)

# Annotate
ax.set_title("Comparing the 2011 plan to an ensemble")
ax.set_ylabel("Democratic vote % (Senate 2012)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()

#### There you go! To build on this, here are some possible next steps:

#### Add, remove, or tweak the constraints
#### Use a different proposal from GerryChain, or create your own
#### Perform a similar analysis on a different districting plan for Pennsylvania
#### Perform a similar analysis on a different state
#### Compute partisan symmetry scores like Efficiency Gap or Mean-Median, and create a histogram of the scores of the ensemble.
#### Perform the same analysis using a different election than the 2012 Senate election
#### Collect Democratic vote percentages for _all_ the elections we set up, instead of just the 2012 Senate election.

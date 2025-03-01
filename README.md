# NK Models

an implementation of NK models and (NK model)[https://en.wikipedia.org/wiki/NK_model] networks and a collection of scripts to analyze them and combine them

for a neuroscience project in a computational biology class

## motivation

for the neuroscience project, were studying consciousness and the modeling of human brains.
one of the labs was based on using NK models as a way to model human neural activity.
we started with using single NK models and analyzing their behavior (usually related to attractors), but then I wanted to do an extension on ways that we can combine NK models to produce patterns that we learned brains do in (A Universe of Consciousness)[https://en.wikipedia.org/wiki/A_Universe_of_Consciousness]. One example of this is reentry, and that is what I tried to model with the selective connections between the NK models.
basically, the final project of this repo is in `combine_nk_models.py`.

as an extension, my teacher challenged me to make more and more complex combinations of the models that exhibit properties of brains as a way to model consciousness and help us (humanity) learn things about the functioning of consciousness.

## files

code files
* `nk_models.py`: all of the implementation code of Nk models. includes evolutionary models and helper functions
  * base nk model
    * count attractors
    * map states to attractors
    * visualization
  * mutable nk model
    * genrating mutations
  * evolutionary model
    * evaluation and the generation of the final fitness score used in evolution
* `testing.py`: self explanatory
* `analyze.py`: also self explanatory
* `robustness_evaluation.py`: use a basic artificial evolution algorithm to evolve NK models to be robust (meaning that attractors dont change much when the model's input paths or functions are changed)
* `cycle_count_frequency.py`: generates a lot of NK models and graphs their count frequency
* `cycle_length_frequency.py`: generates a lot of NK models and  graphs their lengths frenquency
* `combine_nk_models.py`: class implementation and visualization of a network of NK models
* 

generated files
* `Digraph.gv`: an instruction file for the package graphviz for the visualization of a single NK model
  * origin: `NKModel.visualize`
* `Digraph.gv.pdf`: a pdf of the graph that represents the transitions between states of the NK model
  * origin: `NKModel.visualize`
* `evolved_nk_models.txt`: saved weights generated from `robustness_evolution.py`
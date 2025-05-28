# Constraints on inflation from scale-invariant gravity

# WB

Look into lines 145-6 of the `ParamFinderH.py` file

## Abstract:

### Background and context

There is an overwhelming amount of evidence that the Universe underwent a period of accelerated expansion -- inflation -- very early on in its history.  This conclusion is based on careful observations made in the late Universe, combined with physical cosmology and assuming textbook general relativity to correctly describe the gravitational interaction. However, general relativity alone does not provide an inflationary mechanism and (with the possible, and actually quite compelling exception of the Higgs sector) neither does the Standard Model of particle physics.

An interesting possibility is that general relativity descends at low energies from a more fundamental scale-invariant theory, which describes physics at higher energies below the Planck scale. Scale invariance is an attractive constraint from the perspective of model-building, and the additional dynamics that are present in the resulting models are thought to provide a viable inflaton field. So far, the self-interaction potential for such a field has been obtained analytically, but the observable signatures for inflation have not been thoroughly deduced from it.

![Inflation Project](https://github.com/user-attachments/assets/d4b707f2-b61b-4626-ad58-741cbc45fa6c)

### Project objectives

In this project we will investigate the inflationary signatures of scale-invariant gravity using some robust numerical techniques. Depending on the candidate's preferences after the initial exercises are completed, there are two directions to explore:

1. We may focus on the theoretical development of the model, such as the addition of extra so-called "compensator" fields, and the resulting implications for the potential. It is also possible to explore whether unusual features are present in the potential which could give rise to non-standard power spectra (e.g. inflection points leading to ultra-slow-roll inflation).

2. We may go beyond the boilerplate comparison with summary statistics of the Planck data, as usually seen in the [hep-th] literature, and perform an [astro-ph.CO]-style Bayesian analysis to constrain the theory. This route will require the candidate to use high-performance computing techniques and learn about Bayesian inference in precision cosmology.


## Structure of GitHub Repo

Please note, that it may not be up to date and updates are meant to be in shorthand despite the work done.

### LaTeX Folder

The LaTeX folder contains two folders : The project Notebook and the masters thesis folder. 

1. Part III Project Notebook: This includes any updates and current state of progress. This also contains the possible thoughts/ideas I've had regarding how to get from point A to point B. If extended derivations exist in the notebook, this is original and has not been reproduced from elsewhere (there could exist literature on it, for example, the Weyl curvature scalar); that is, this has been arrived at independently. I have arrived at some particular formulae that I have not found any resources elsewhere. (
Please note, it may not be up to date and updates are meant to be in shorthand despite work done.)
2. Part III Master's Thesis: This is the first draft (and through time, the final) version of my findings-formulae, graphs, and experimental matching- work on it has begun on 14th March 2025, following a meeting with Dr. Barker and Dr. Giorgos, and realizing there is enough data to write the thesis. This might also be the format for publication.
   
### Mathematica Folder

1. The "Learning Notebook.nb" serves as an introductory testfile + playground for me to learn the commands used in the rest of the notebooks. The "Salvio into Barker variables.nb" too serves as an elementary application of these tests. These may be overlooked for other viewers.
2. The "Finding Conf Invariant Tensor.nb" was my attempt at deriving a conformally invariant version of the Ricci Scalar. This was before I learned of the existence of the Weyl Curvature Scalar, but either way, served as an excellent testfile for me to learn and practice the xAct module.
3. The "Finding Action.nb" is my ongoing attempt at solving the EOM for the B vector and entering that into the action to find the final form of the potential. Further work needed in this section (Is it possible to simplify further on Mathematica at all?).
4.  The "Rubi Learning.nb" is the nb in which I calculated the field redefinitions and tried out possible different expansions for field redefinitions /potential redefinitions
5. "Numerical Integrator.nb" is the file being used to crosscheck python's numerical integrator to get the canonical field.
   
### Python Folder

1. The figures folder contains important figures generated which get fed directly into the LaTeX figure calls.
2. The files "Approximate Potential.py", "Potential Solver.py", and "Translating Variables.py" are the Michaelmas/winter workings. These are mostly to familiarize myself with the methods and apparatus required for the analysis of the actual project. These consist of analysis of the Barker/Silvio potentials and serve to understand the slow roll parameters and numerical methods needed to work with the extended model I formulated in the Lent term.
3. The files "PPotenial.py" and "PPotenial V1.5.py" are to analyse the different regimes of the extended model (plateau and hilltop), there are other scenarios which have not been uploaded too.
4. The file "Full field as func of N.py" is an exact solver of the field $\varphi$ vs N (the efolding time). The differential equation was derived independently (more details in lab notebook) and have been numerically solved to get the relevant graph in the Figures folder.
5. The file "Potential Solver for noncanonical action.py" plots the potential of the canonical field without making approximations in contrast to the files in 3. There are also attempts to plot the slow roll parameters here.

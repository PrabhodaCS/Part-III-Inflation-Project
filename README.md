# Tasks 

# Things we need for a nice paper

- Star plot on the Planck legacy showing how the values of each coupling alter the ns/r.
- Compute the running of the spectral index.
- Possible transition to polychord (?)
- Possible transition to MS integration (?)
- Figure out the theory... expect to spend most of project thinking about this.
- Weyl anomaly (talk to Carlo Marzo)

Example of variational derivative in xAct (let's say you have metric `G`, covariant derivative `CD`, and indices `a`, `b`):
```Wolfram
Expr=<some scalar-valued expression in xAct>
Expr//Print;
Expr//=VarD[G[a,b],CD];
Expr//Print;
```
Carefully check by testing some examples whether the factor of `Sqrt[-Det[G]]` is assumed in the result.

# WB

- Get PCS set up on Newton.
- Maybe try to parallelize the optimizer.
- Think about Hertzberg argument.

# PCS

- Delete all useless files, try to consolidate scripts as much as possible, have meaningful file names.
- Get the optimizer working in parallel, spend no more than 1-2 days getting it fast.
- If still you have problems, let WB have a go.
- Learn about ssh (within Windows).
- Recompute the JP current analysis in xAct, then extend to our model.

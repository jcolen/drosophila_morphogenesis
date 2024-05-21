# Mutant analysis in response to reviewer comments

To test our machine-learned model, we will examine mutant embryos in the following folders

- flydrive/Halo_twist[ey53]
- flydrive/toll[rm9]
- flydrive/spaetzle[A]

In **VisualizeMutantData.ipynb**, we plot the mutant datasets and examine time-alignment for embryos like twist that show development of a DV gradient.

In **FlowNN.ipynb**, we demonstrate the ability to predict flow from mutant myosin fields using a deep neural network.

In **MaskedPCAModel.ipynb**, we show how mutant myosin patterns differ from WT in the principal component space.

In **ClosedLoop_NeuralODE.ipynb**, we forecast the coupled dynamics of myosin, eCadherin, and tissue flow using the equations learned in Step 02 of this pipeline.

In **ClosedLoop_PCAFlow.ipynb**, we show that a nonlinear kernel ridge regression model that maps between PCs of myosin and flow can achieve reasonable accuracy at dynamics forecasting in mutants.
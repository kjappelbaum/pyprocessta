# pyprocessta

> Making the analysis of data from chemical engineering processes boring again!

`pyprocessta` is a suite of tools that aims to help chemical engineers with the analysis of their time series. `pyprocessta` provides tools for the following steps

- exploratory data analysis
- data cleaning
- time series modeling
- causal impact analysis

Note that this package is still in an early, experimental, development phase.

## Installation

To install the development version on a Unix system (we did not test on a Windows machine) use

```bash
pip install git+https://github.com/kjappelbaum/pyprocessta.git
```

The installation should be completed within minutes or seconds.

Note that the package currently depends on our fork of the darts package and is not tested on the latest official release of the darts library.

## Reproducing the analysis in "Deep learning for industrial processes: Forecasting amine emissions from a carbon capture plant"

For the notebooks and scripts we used in this work, see the `paper` directory. This directory also contains a "freeze" of the conda environment we used in our work.

You can use the notebooks in this directory as an example of how to use the library.
The use of the main functionalities is also described in the documentation.

## Acknowledgments

This project was supported by the PrISMa Project (No 299659), which is funded through the ACT programme (Accelerating CCS Technologies, Horizon2020 Project No 294766). Financial contributions made from: Department for Business, Energy & Industrial Strategy (BEIS) together with extra funding from NERC and EPSRC Research Councils, United Kingdom; The Research Council of Norway, (RCN), Norway; Swiss Federal Office of Energy (SFOE), Switzerland; and US-Department of Energy (US-DOE), USA, are gratefully acknowledged. Additional financial support from TOTAL and Equinor, is also gratefully acknowledged.

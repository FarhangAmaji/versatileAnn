# Quick Summary

This project aims to provide **utility tools** that **facilitate** the **development of models**, particularly in **deep learning**, making **prototyping models faster** and saving significant **development time**. It includes the latest **state-of-the-art models** in forecasting and **prepared datasets** for model testing and **benchmarking**, catering to both **model experimentation** and robust development environments for **AI researchers**.

### Who this project is for?!

- `anyone who uses Pytorch (AI deep learning library)` or `similar`(any other deep learning framework)
- `sequential data (Forecasting & NLP & audio) AI researchers`
- For people who want to use `ready forecasting models` 

## **Project Features**

- **Simplifies Complex Processes:**

  `Bundles` `commonly used functionalities`, `streamlining` complex processes into a `single function call` while maintaining `customizability`.

- **Reduces Development Time:** 

  `useful ready functionalities` significantly `impact` development `timelines`. Therefore improves `Development & User Experiences`. Enabling development with `less boilerplate code`.

- **Error Detection and Prevention:**

  It also `raises errors` and `warnings` to `prevent many potential errors`; this is especially `very needed` when `working` with `Python` and Python packages.

### brazingTorch

- **Streamlined Deep Learning Pipeline:** Simplifies model development with a customizable pipeline.
- **Automatic Model Architecture Differentiation and Tracking:** `Prevents losing promising models` by automatically detecting model architectures.
- **Resettable Dataloader, Optimizer, Model:** `not supported in PyTorch`.
- **Automatic preRunTests:** Automatically runs `sanity checks` and `optimizations` including FastDevRun for `pipeline errors`, overfitBatches for quick sanity checks, Profiler for `bottlenecks` with TensorBoard charts, and `Learning Rate & Batch Size Optimization`.
- **Simplified Experience with Smart Defaults:** Includes `general regularization`, `best model save` on best validation loss, `learning rate warm-up`, `early stopping`, `gradient clipping`, and `automatic GPU device setting`, with informative messages for customization.
- **Regularization:** Supports `regularization on each layer` and `L1 regularization`, `not supported in PyTorch`.
- **Special Modes:** Provides ready pipelines for `dropoutEnsembleMode` and `variational autoencoder mode`, which can be used together.

### dataPrep & commonDatasets

- **Specialized time series dataset:** `Efficient` `easy` `sequence generation` with `error prevention`, faster than pandas, supports `nested dictionaries` for `complex data`, and `optimizes GPU memory usage`.
- **Time Series Splitting:** Splits time series data `normal` and `shuffled`, `considering` the `sequential nature`. Supports "NSeries"(`multiple time series` reside `within a single dataframe`) data splitting.
- **Specialized Normalizers:** `Numerical and label normalizer` for `single and multiple columns`, also `supports "NSeries" data`.
- **Exogenous Variables Support:** Handles various exogenous variables, both `static` and `time-varying`, `known or unknown in the future`.



This project is extensive, with over 15,000 lines of code and rigorously tested with over 400 tests. It offers a wealth of **unique and innovative functionalities and options**, detailed in the main README.md — [**worth exploring further**](https://github.com/FarhangAmaji/versatileAnn).

# Quick Summary

In some subfields of AI like NLP and computer vision, `foundational models` have become dominant, streamlining the development process. This dominance is largely `due to the intrinsic characteristics` of the data in those fields, which make it well-suited to use foundational models. However, for `more general AI tasks`, where the `variety and complexity of data` types are `much higher`, suitable `foundational models either don't exist or cannot be effectively applied`. As a result, there is a growing need for flexible tools that facilitate `model development from scratch`, especially when navigating the vast landscape of possible architectures and hyperparameters.

This project addresses that need by offering utility tools for `hyper-model search`. These tools significantly `accelerate the prototyping` process, `saving valuable development time` while ensuring that the search for optimal models is more reliable. Through the use of `smart defaults`, intelligent `warnings`, and automated `checks` for common `errors`, the framework `minimizes` the likelihood of critical semantic mistakes that can `derail` projects late in development. This leads to a smoother, more assured development process where the `risk of failure` is greatly `reduced`.

Moreover, the project includes a core framework that supports any PyTorch model, suitable for more general AI users, along with `specialized forecasting libraries` packed with essential tools for `pre-processing and post-processing time-series` data. These libraries provide access to the latest `state-of-the-art models in forecasting`, alongside curated `datasets for testing and benchmarking`. Whether you're developing a new model from scratch or experimenting with existing cutting-edge forecasting techniques, this framework offers a robust, ready-to-use environment for AI research and experimentation.

### Who this project is for?!

- `General AI users`: anyone who uses `Pytorch` (AI deep learning library) or `similar` (any other deep learning framework)
- `AI researchers` especially who work with sequential data (Forecasting & NLP & audio) models
- `Practitioners` and `enthusiasts` who want to `experiment` with or implement ready-made state-of-the-art forecasting models for their use cases

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



This project is extensive, with over 18,000 lines of code and rigorously tested with over 400 tests. It offers a wealth of **unique and innovative functionalities and options**, detailed in the main README.md — [**worth exploring further (project itself)**](https://github.com/FarhangAmaji/versatileAnn).

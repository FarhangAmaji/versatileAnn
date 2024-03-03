# what versatileAnn is

While developing machine learning models can be time-consuming, even the smallest typos in programming can derail months of hard work, rendering unreliable results. Utilizing `well-tested tools` can significantly improve the experience of both using and developing models.

This project aims to provide utility tools that streamline the development of models, particularly those in the realm of deep learning. As the name `"VersatileAnn"` suggests, the project strives to be beneficial across different deep learning subfields, including natural language processing (NLP), computer vision, forecasting, and other more general use cases. Not only can it be useful for people to `simply try out some models`, but it's designated to be especially handy for `AI researchers` who require a robust and efficient development environment.

The project comprises 5 distinct sections: `dataPrep`, `commonDatasets`, `brazingTorch`, `models`, and `machineLearning`. `BrazingTorch` and `machineLearning` offer valuable resources for individuals aspiring to delve into development and utilizing deep learning and machine learning approaches, respectively.

However, the `models`, `dataPrep` and `commonDatasets` sections are currently `specialized` for `forecasting`. `models` contains many of forecasting `state of the art` models. It's worth noting that NLP falls under the category of models dealing with `sequential data`, where information is processed in a specific order. This alignment allows VersatileAnn to be particularly useful for `NLP models`. The project holds potential for incorporating novel ideas similar to the existing ones, but these are yet to be implemented for other subfields.

kkk this project has really some cool unique features which don't exist elsewhere.

Additionally, VersatileAnn aims to provide an easy-to-use tool for testing various machine learning models with different hyperparameters. This functionality can significantly save development time and effort by streamlining the process of identifying the optimal configuration for a specific model.

## sections

Note each section has its own **readme.md**.

### brazingTorch

`brazingTorch` was developed here in this project, but as it offers valuable functionalities for the broader deep learning community, so it was moved to its own github repo: https://github.com/FarhangAmaji/brazingTorch

it can be found at `/brazingTorchFolder` .

#### Cool features:

- Note many features mentioned below have their options and may be turned off.

1. **Streamlined Deep Learning Pipeline:**

   - BrazingTorch `simplifies` model development by providing a `customizable pipeline`.
   - In `most cases`, you only need to define the forward pass and call the `fit` method, making it remarkably user-friendly.
   - For further customization, the `commonStep` method can be overridden to suit specific needs.

2. **Automatic Model Differentiation:** 

   - This feature `automatically` `detects` and `names` your `model architecture`, even with minor changes in lower layers.
   - Assume you're building a model to closely match your data. While the process can be messy, it's not uncommon to encounter promising models during development. Unfortunately, `without proper documentation`, we might forget which model yielded these promising results. This can lead to either losing the model entirely or wasting time rediscovering it.
   - Note when the architecture is named so it can be easily distinguished in tensorboard charts.
   - Additionally, the architecture details are saved in `architecture.pkl` for easy reference.

3. **resettable** `dataloader`, `optimizer`, `model` :

   - these are unique features and cannot be found elsewhere
   - Allows you to easily modify the **batch size** of data loaders, offering greater flexibility during training. this is impossible to do in PyTorch.

4. **preRunTests**

   - preRunTests is a  default set of models runs, to provide useful sanity checks and details. If preRunTests hasen't been executed already, is going to be executed automatically before the `.fit` method.

   - 1. **FastDevRun:** This runs the entire pipeline, `checking` for any `errors` that might cause the `whole process` to fail. This allows you to start the training, walk away, and come back later without worrying about unexpected failures.
     2. **overfitBatches:** This test trains the entire model on a single batch, a popular technique for quick `sanity checks` and shows model's ability to provide good result.
     3. **Profiler:** This run helps identify `potential bottlenecks` in the training pipeline. It provides a TensorBoard visual chart showing the time consumption of each step. While it's usually hard for people to make a this chart, but in this project is provided effortlessly.
     4. **Learning Rate & Batch Size Optimization:** This set of tests automatically finds the optimal learning rate and batch size for your data loader, improving training efficiency.

5. **Useful options by default**

   - Other than making the pipeline easier, this project add lots of useful default so in the case that if you have forgotten to utilize these features you may get reminded to use them. Of course, you still have the flexibility to turn them off if needed.

   1. General regularization (applied on all layers)
   2. Best Model save on best `val` loss
   3. Learning rate warm up
   4. Early stopping
   5. Gradient clipping

6. **Regularization:**

   - In `tensorflow` regularization can be applied on each layer, but pytorch doesn't support this. also pytorch doesn't support `L1` reguarization. but these are available in this project.

7. **Special modes:**

   - It provides customized pipelines to utilize `dropoutEnsembleMode` and `varational autoencoder mode`, even can used together.

8. **Autosetting device**:

   - This project extends its compatibility to include **MPS devices**, recently introduced by PyTorch for Apple M1 processors. Previously, PyTorch on Mac solely relied on CPUs. It's important to note that MPS devices **currently lack support for 64 decision data types**, and this project has incorporated necessary considerations to address this limitation.

9. **standalone model load:**

   - PyTorch Lightning typically just saves model parameters. This project uniquely saves the **class definitions** in the checkpoint, enabling direct loading **without requiring the original model code**. This provides a standalone, self-contained model.

10. **Easier optimizer/scheduler setup:**

    - Dedicated properties simplify configuration compared to PyTorch Lightning.

11. **Pipeline helper functions**: Offers some methods to work easier in framework format

12. **Color Coding for Clarity: Warnings, Errors, and Infos**



brazingTorch is a high level wrapper around `pytorch lightning`, which itself is a wrapper around `pytorch`. therefore, you can seamlessly integrate BrazingTorch with other PyTorch, PyTorch Lightning, and even TensorBoard features for a tailored deep learning experience.

note this project only utilizes pytorch lightning, just for the case the users wants to use `multiple gpus` or `distributed computations` provided by pytorch lightning, otherwise all pipeline could have designed from scratch. so finally noting that `multiple gpus` or `distributed computations` are also supported.

This project utilizes PyTorch Lightning primarily for its `multi-gpu` and `distributed computing` capabilities, otherwise, the entire pipeline could have been built from scratch. so it can be understood brazingTorch supports this.

brazingTorch is designed as a **framework**, following a common approach for building pipelines like this. Similar to PyTorch Lightning, it offers a structured and reusable foundation for building and managing the components involved in the pipeline.

---

### dataPrep & commonDatasets

- A collection of functions named `dataPrep` is available to help `prepare time series data` for modeling purposes. These functions are located in the `\dataPrep` directory.
- `\commonDatasets\commonDatasetsPrep` directory contains functions that leverage the `dataPrep` functionalities. These functions provide `ready preprocessed` data of some famous time series datasets, along with their dataset and dataloader objects.

#### features

- **Powerful Tools for Time Series Data Preparation**

  - There are lots of useful functions in `\dataPrep\utils.py` which help to transform raw data and do data preparation operations.

- **Time Series Splitting:**

  - Offers functions to split time series data into training, validation, and testing sets, even shuffled.
    - Can `split into three sets` (train, validation, and test) instead of the usual two, making the splitting much easier. Note the most important thing here is that it `considers the sequential nature of time series data`, crucial for accurate preparation.
  - Includes a dedicated function for splitting `"NSeries"` timeseries data, where multiple time series reside within a single dataframe along each other in different rows.

- **Various specialized normalizers:**

  - This project utilizes three main types of normalizers, each with a subtype for handling numerical and label data:

  - also can easily `fit`, `transform` and `inverseTransform` the `all columns` or `single column` of the dataframe.

    1. **SingleColNormalizer:**

       - This is the standard normalizer for individual columns.

    2. **MultiColNormalizer:**

       - This normalizer treats multiple columns as one during normalization.

       - It calculates statistics (mean, classes, etc.) across all specified columns for normalization.

    3. **MainGroupSingleColNormalizer:**

       - This specialized normalizer is designed for "NSeries" data.
       - NSeries refers to data containing multiple distinct time series within a single column.
       - The normalizer assigns and manages a separate `SingleColNormalizer` for each unique group within the NSeries data. This ensures proper normalization for each individual time series.

- **`VAnnTsDataset` specialized time series dataset:**

  - Utilizes `parallel` capabilities of pytorch dataset

  - **Efficient sequence generation with `getBackForeCastData`:**

    The `getBackForeCastData` function offers a convenient way to generate different types of sequences from pandas DataFrames:

    - **Cast modes:** The function supports various modes for generating sequences, including backcast, forecast, fullcast, and singlePoint.
    - **Faster than pandas:** Compared to using pandas DataFrames directly, `getBackForeCastData` is specifically designed to utilize a wrapper around pandas dataframe but making it faster to fetch data.
    - **Accuracy:**  So many errors that can occur during the sequence generation process but it helps mitigate those errors, resulting to save time and effort.
    - **Lots of other options:**
      - 

  - **Efficient Data Handling with nested dictionaries:**

    - **Easy Data Structuring:** While using dictionaries is an intuitive way to classify your data, this dataset makes it even easier. You can structure your data using dictionaries, and for more complex data, you can even use nested dictionaries (dictionaries within dictionaries).
    - **Automatic Handling of Dictionaries:** Forget manual manipulation! This dataset automatically detects whether your data is in dictionaries or nested dictionaries. It handles these structures for loss calculations and even converts them to tensors for your model.
    - **GPU Memory Efficient Tensor Conversion:** This dataset prioritizes memory efficiency. It only converts your data to tensors at the final step, right before feeding it to the model. This avoids unnecessary GPU memory usage and optimizes allocation.

- `dataprep` is designed as a `library` that requires you to manually apply its functions to your data. While some other packages offer automated data preprocessing, researchers often need to control feature selection and specify what operations to be applied on each feature. Automating this step can create a `"black box"` effect, where you lose understanding of the data preparation process. This lack of transparency can be detrimental for AI researchers who need to have a deep grasp of the preprocessing steps.

---

kkk brazingTorch is a framework but dataprep libray making it not a blackbox ; as AI researcher really need to know what exactly is going on on the data preprocessing steps, or model

kkk good clean code practices focusing on readability. kkk public and private mthod/class naming

kkk cccUsage

kkk good detailed comments with numerating them by importance; add colors

kkk this is a big project more than 15,000 lines of code. so please be patient to get familiar with this project. but it can be guessed with cool features and features to prevent errors, it really worth it to invest some time to it.






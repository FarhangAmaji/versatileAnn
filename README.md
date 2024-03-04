## Table of Contents

- [**Who this project is for**](#whoThisProjectIsFor)**
    - [what is forecasting](#whatIsForecasting)
- [**What versatileAnn is**](#whatVersatileAnnIs)
- [**sections features**](#sectionsFeatures)
  - [brazingTorch](#brazingTorchFeatures)
  - [dataPrep & commonDatasets](#dataPrepNCommonDatasetsFeatures)
- [**Project additional infos**](#additionalInfos)
- [**How to get started**](#HowToGetStarted)
  1. [use for timeseries data preparation](#useForTsDataPrep)
  2. [use BrazingTorch](#useBrazingTorch)
  3. [models](#models)
  4. [examples](#examples)

  5. [machineLearning](#machineLearning)

  6. [projectUtils](#projectUtils)
# Who this project is for?!<a id='whoThisProjectIsFor'></a>

This project has some sections and `"who this project is for"` `depends` on what `section` you may use.

- anyone who uses pytorch (`brazingTorch` section)
- for people who want to use ready forecasting models (whole project)
- forecasting AI researchers (whole project)

# what versatileAnn is

While developing machine learning models can be time-consuming, even the smallest typos in programming can derail months of hard work, rendering unreliable results. Utilizing `well-tested tools` can significantly improve the experience of both using and developing models.

This project aims to provide utility tools that streamline the development of models, particularly those in the realm of deep learning. As the name `"VersatileAnn"` suggests, the project strives to be beneficial across different deep learning subfields, including natural language processing (NLP), computer vision, forecasting, and other more general use cases. Not only can it be useful for people to `simply try out some models`, but it's designated to be especially handy for `AI researchers` who require a robust and efficient development environment.

The project comprises 5 distinct sections: `dataPrep`, `commonDatasets`, `brazingTorch`, `models` and `machineLearning`. `BrazingTorch` and `machineLearning` offer valuable resources for individuals aspiring to delve into development and utilizing deep learning and machine learning approaches, respectively.

However, the `models`, `dataPrep` and `commonDatasets` sections are currently `specialized` for `forecasting`. `models` contains many of forecasting `state of the art` models. It's worth noting that NLP falls under the category of models dealing with `sequential data`, where information is processed in a specific order. This alignment allows VersatileAnn to be particularly useful for `NLP models`. The project holds potential for incorporating novel ideas similar to the existing ones, but these are yet to be implemented for other subfields.

Additionally, VersatileAnn aims to provide an easy-to-use tool for testing various machine learning models with different hyperparameters. This functionality can significantly save development time and effort by streamlining the process of identifying the optimal configuration for a specific model.

This project is quite extensive, with over 15,000 lines of code. While this summary highlights some key features, there are many `more unique functionalities and options` available.  These features are not only `well-tested` with more than 300 unit tests, but also designed to `save` you significant `development time`.  By incorporating these functionalities, you can `avoid errors` through the built-in error handling mechanisms. For a deeper dive, `each section` within the project has its own dedicated **readme.md** file explaining these features in detail.  Additionally, these `sections` are designed to `work together` seamlessly, leveraging each other's functionalities.

## sections

### brazingTorch

`brazingTorch` was developed here in this project, but as it offers valuable functionalities for the `broader deep learning community`, so it was moved to its own github repo: https://github.com/FarhangAmaji/brazingTorch

it can be found at `/brazingTorchFolder`.

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

  - There are lots of useful functions in `\dataPrep\preprocessing.py` which help to transform raw data and do data preparation operations.

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

  - **Efficient Data Handling with nested dictionaries:**

    - **Easy Data Structuring:** While using dictionaries is an intuitive way to classify your data, this dataset makes it even easier. You can structure your data using dictionaries, and for more complex data, you can even use nested dictionaries (dictionaries within dictionaries).
    - **Automatic Handling of Dictionaries:** Forget manual manipulation! This dataset automatically detects whether your data is in dictionaries or nested dictionaries. It handles these structures for loss calculations and even converts them to tensors for your model.
    - **GPU Memory Efficient Tensor Conversion:** This dataset prioritizes memory efficiency. It only converts your data to tensors at the final step, right before feeding it to the model. This avoids unnecessary GPU memory usage and optimizes allocation.

- `dataprep` is designed as a `library` that requires you to manually apply its functions to your data. While some other packages offer automated data preprocessing, researchers often need to control feature selection and specify what operations to be applied on each feature. Automating this step can create a `"black box"` effect, where you lose understanding of the data preparation process. This lack of transparency can be detrimental for AI researchers who need to have a deep grasp of the preprocessing steps.

---

### Project additional infos

- This project prioritizes `code clarity` and maintainability by adhering to best practices for `clean code`. A clear `naming convention` is implemented for `public and private functions`, methods, and classes. Private elements, denoted by leading underscores (`_`), are not intended for direct user interaction. While technically usable, it's advisable to exercise caution when utilizing them. This naming convention simplifies project navigation, allowing you to readily `identify` the `primary functionalities` and guiding you towards the appropriate functions, methods, or classes to get started.

- This project utilizes comments extensively to enhance code readability and understanding. Comments tagged with `cccUsage` offer specific guidance on how to `leverage` the project's `functionalities`. Additionally, comments numbered `ccc1` to `ccc4` indicate their relative `importance` or the `scope they influence`, aiding in navigation.

- To further enrich your coding experience, consider incorporating `colored comments` into your development environment. The project provides references for these color-coding conventions:

  - **PyCharm**: `\devDocs\conventions\convention_todoTypes.py`

  - **VSCode** better comments extension *settings.json*: `\devDocs\conventions\conventions.md`

- Also on different comments it's been indicated that in more comprehensive explains exist in `\devDocs\codeClarifier` files.

- to have same settings specially the formatter used in development of this project, in `pycharm` import `\devDocs\versatileAnn_pycharmFormatter.editorconfig`.

- also reading tests in `\tests` folder can be useful in order to understand the project more.

# How to get started

in order to know to get started, I should know what do yo want to do and what uses cases of this project do you want to use?!

1. **use for timeseries data preparation:**

   - note depending on how you understand and learn new things, you may go from 1 to 5 in list below, or you may only start at 5 (commonDatasetsPrep) which utilizes the first four.

   1. take a deep look at public functions in `\dataPrep\preprocessing.py` which are main functions for core data preprocessing operations to manipulate the raw data. getting familiar with these functions would help you a lot, so you will never do recreate those helper functions.

   2. `\dataPrep\dataset.py`

      for beginning and also for most cases you may only need to know:

      1. how to inherit from `VAnnTsDataset` and override its `__getitem__` method to create a simple dataset.
      2. for easier sequence generation you need get to know `getBackForeCastData` method and its options
      3. `getBackForeCastData_general` is also available which gives a bit of more freedom comparing to getBackForeCastData

   3. `\dataPrep\dataloader.py`

      - mostly there is nothing to do here by you and all things are handled automatically and you need to create an instance of it with dataset and of course may use its other options.

   4. located at `\dataPrep\normalizers`, as mentioned above there are 3 sets of normalizers:

      - note as these 6 normalizers share some similar responsibilities they share some `Interfaces` with each other.

      1. `SingleColStdNormalizer` and `SingleColLblEncoder` at `singleColNormalizer.py`
      2. `MultiColStdNormalizer` and `MultiColLblEncoder` at `multiColNormalizer.py`
      3. `MainGroupSingleColStdNormalizer` and `MainGroupSingleColLblEncoder` at `mainGroupNormalizers.py`

   5. take a look at files at `\commonDatasets\commonDatasetsPrep`, there are several files each doing data preprocesses on raw data of some famous datasets also providing their dataset and dataloaders. note this understanding these files means you understand whole of the `dataPrep` section.

2. **use BrazingTorch:**

   - it's a better alternative for pytorch and pytorch lightning and you may need:
     1. take a look at arguments at `\brazingTorchFolder\brazingTorch.py`
        - note there it's been addressed where implementation of each arguments is.
        - note for better `separation of concerns`, most of their implementations are at files in `\brazingTorchFolder\brazingTorchParents`. so reading these files would help understand how exactly this package works.
        - also there are some custom layers at `\brazingTorchFolder\layers\customLayers.py`
        - and custom callbacks at `\brazingTorchFolder\callbacks.py`

3. **models:**

   - If you want get familiar to some latest `state of the art` models in forecasting you may take a look at `\models`. it contains the code of those models with compatibility with BrazingTorch.

4. **examples:**

   - there are examples on how to define models which use prepared data of `\commonDatassets\commonDatasetsPrep` or some examples have random data. then either use the models in `\models` or create a `custom model`.

5. **machineLearning:**

6. **projectUtils:**

   - complementary functions are at `\projectUtils`
   - you may look at files at `\projectUtils\dataTypeUtils` and also `projectUtils/misc.py`

---

- note currently `\models` don't work as their not yet changed to work with newer version of brazingTorch
- examples are not workings
- machineLearning has not been committed

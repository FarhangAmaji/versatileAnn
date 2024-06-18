## Table of Contents
- [**Who this project is for**](#whoThisProjectIsFor)
- [**What versatileAnn is**](#whatVersatileAnnIs)
- [**Project Features**](#projectFeatures)
    - [**Sections**](#sections)
        - [brazingTorch](#brazingTorchFeatures)
        - [models](#modelsCurrentlySupported)
        - [dataPrep & commonDatasets](#dataPrepNCommonDatasetsFeatures)
        - [Machine Learning](#machineLearning)
- [**Project additional infos**](#additionalInfos)
- [**How to get started**](#HowToGetStarted)
  1. [use for timeseries data preparation](#useForTsDataPrep)
  2. [use BrazingTorch](#useBrazingTorch)
  3. [models](#models)
  4. [examples](#examples)
  5. [machineLearning](#machineLearning)
  6. [projectUtils](#projectUtils)
- Appendix
    - [what is forecasting](#whatIsForecasting)

![Logo](https://github.com/FarhangAmaji/versatileAnn/blob/master/BrazingTorch.jpg)

## quickSummary

This readme is quite detailed, so it's better to take look at [quickSummary](https://github.com/FarhangAmaji/versatileAnn/blob/master/quickSummary.md).

## How to read this!?!?

This is a `quite big project`, and this readMe is `a bit long`, so `recommended` for the `first time` :

- `only read` `highlights` and **bold texts** and just get familiar with features this project has.

- `avoid additional infos`. They can help you to understand what's going on in the project.

- Also take a look at "**super quick glimpse at files**" in ["**How to get started**"](#HowToGetStarted).

# Who this project is for?!<a id='whoThisProjectIsFor'></a>

This project firstly is for `AI enthusiasts`, if you are `interested` in `AI`, but `Forecasting` and `NLP` (natural language processing) are `not` in your fields of `interests` in AI, then you may go straight to https://github.com/FarhangAmaji/brazingTorch

- [in the case you don't know what **forecasting** is, click here](#whatIsForecasting)



This project has some sections and `"who this project is for"` `depends` on what `section` you may use.

- `anyone who uses Pytorch (AI deep learning library)` or `similar`(any other deep learning framework) ([brazingTorch section](#brazingTorch)):
  
  if you are using pytorch, or any other deep learning framework like `tensorflow`, `keras`, `caffe`, etc, it worths to `try` **"brazingTorch"** at `least once`.
  
  - **Very Powerful, Unique Features:** offers `unmatched powerful`, `unique features` compared to traditional frameworks like PyTorch, PyTorch Lightning, etc.
  - **User-friendly Customizable Pipeline:** Pre-built pipeline `saves` you `significant development time` by `minimizing coding`, letting you focus on your model, and is also `customizable` with ease.
  - **Automatic Model Architecture differentiation and tracking:** `No need to worry` about naming or `losing promising models`.
  - **Resettable Components:** Unique resettable `dataloaders`, `optimizers`, and even can reset `models`.
  - **PreRunTests:** Get `sanity checks` and `optimize training efficiency` automatically.
  - **Smart defaults with reminder warnings:** `Simplify your workflow` and customize as needed.
- For people who want to use `ready forecasting models` (whole project)
  - **Code of latest `state of the art` (`cutting edge models of AI, publicly published in journals, achieving the best results`) models in `forecasting`**
  - **ready processed datasets:** ready processed data for some famous timeseries datasets.
- `sequential data (Forecasting & NLP & audio) AI researchers` (whole project)
  
  - **Rich and Versatile Data Processing and Analysis Library**
  - `Not Reinventing the Wheel`, helps to `avoid` to develop from `scratch`.
  - Helps to `avoid errors`.
  - supports `variety of data types` including exogenous in processing and model components
  - Has some `unmatched unique features`, which provide greater flexibility for your workflow.
- **Developers Interested in Innovative Projects**


---

# What versatileAnn is<a id='whatVersatileAnnIs'></a>

This project aims to provide `utility tools` that `facilitate` the `development of models`, `particularly` those in the realm of `deep learning`, making `prototyping models faster`, designed to `save` you significant amount of `development time`. making it well-suited for AI development compared to other similar packages.

This project is `quite extensive`, with over 15,000 lines of code. While `this readMe` highlights some `key features`, there are many `more unique novel functionalities and options` available.

These features are not only `well-tested`, with more than 400 unit tests, also with `smart errors` `preventing possible semantic mistakes`, making your `development reliable`.

Beside has latest the `state of the art models` in the `field of forecasting`, and also `already prepared datasets` so you can test and `benchmark` models you are developing. Not only can it be `useful for` people to `simply try out some models`, but it's `designed` to be `especially handy` for `AI researchers` who require a robust and efficient development environment.

# **Project Features**<a id='projectFeatures'></a>

- **Simplifies Complex Processes:**

  `Bundles` `commonly used functionalities`, `streamlining` complex processes into a `single function call` while maintaining `customizability`.

- **Reduces Development Time:** 

  `useful ready functionalities` significantly `impact` development `timelines`. Therefore improves `Development & User Experiences`. Enabling development with `less boilerplate code`.

- **Error Detection and Prevention:**

  It also `raises errors` and `warnings` to `prevent many potential errors`; this is especially `very needed` when `working` with `Python` and Python packages.

- **Enhances Code Reliability:** 

  In programming `even minor typos` can `waste months of hard work`. `Well-tested tools` help `minimize errors` and `ensure reliability` of code.

- **Elevates Code Quality:**

  The essence and attention to detail ingrained in this project `guide` your development towards `cleanliness`, meticulously addressing even minor details to `prevent errors` and enhance code robustness.

- **Enhances Future Development:**

  From the perspective of further developing this project, it's been built with `clean code principles` and a tidy `structure`, simplifying `future developments `. Additionally, it's adorned with `detailed internal comments` for easy understanding and modification.

- **Compatible Integration with Popular Libraries:**

  Compatibility with pandas, numpy, scikit-learn, PyTorch, and PyTorch Lightning `allows` for `integrating` of `your existing works` into this project with low effort.

- **Exciting Features and Cool Ideas:**

  This project is packed with lots of fresh features and awesome ideas, sparking creativity and innovation.



## Sections<a id='sections'></a>

**sections:** The project comprises 5 distinct sections: `dataPrep`, `commonDatasets`, `brazingTorch`, `models` and `machineLearning`.

- **`BrazingTorch` and `machineLearning` sections:** Offer valuable resources for individuals aspiring to delve into `development` and utilizing `deep learning` and `machine learning` approaches, respectively. and are `suitable` for `majority of machine learning community` and can be `used by anyone who uses pytorch` or `deals with machine learning models`.

- **`models`, `dataPrep` and `commonDatasets` sections:** But these sections are currently `specialized` for `forecasting`. `models` contains many of `forecasting` `state of the art models`. It's worth noting that `NLP` and `AI audio applications`, `alognside forecasting`, falls under the category of models dealing with `sequential data`, where information is processed in a specific order. This alignment allows versatileAnn to be also useful for `NLP models`. 



### brazingTorch<a id='brazingTorchFeatures'></a>

brazingTorch is a **high level wrapper** around `pytorch lightning`, which itself is a wrapper around `pytorch`. therefore, you can seamlessly `integrate` BrazingTorch with other `PyTorch`, `PyTorch Lightning`, and even TensorBoard features for a `tailored` deep learning experience.

`brazingTorch` was developed here in this project(versatileAnn), but as it offers valuable functionalities for the `broader deep learning community`, so it was moved to its own github repo: https://github.com/FarhangAmaji/brazingTorch

it can be found at `\brazingTorchFolder`.

#### Cool features:

- Note many features mentioned below have their `options` and `can be turned off`.

1. **Streamlined Deep Learning Pipeline:**

   - BrazingTorch `simplifies` model development by providing a `customizable pipeline`.
   - In `most cases` for simple architectures, you may `only` need to `define` the `forward method` and call the `fit` method, making it remarkably user-friendly.
   - For further `customization`, the `commonStep` method can be `overridden` to suit specific needs.

2. **Automatic Model Architecture Differentiation and Tracking:** 

   - This feature `automatically` `detects` and `names` your `model architecture`, even with minor changes in lower layers.

   - **No need to worry about losing promising models:** 

     Model `development` can become `messy` and documentation may be forgotten or corrupted, leading to the risk of `losing promising models` or `spending hours rediscovering` them; this feature `mitigates such risks`.

   - Note when the `architecture` is `named` so it can be easily `distinguished` in `tensorboard` charts.

   - Additionally, the architecture details are saved in `architecture.pkl` for easy reference.

3. **Resettable** `dataloader`, `optimizer`, `model` :

   - these is `unique` unmatched feature compared to traditional libraries like PyTorch, PyTorch Lightning, etc
   - Allows you to easily modify the **batch size** of data loaders, a functionality not available with PyTorch dataloader.

4. **Automatic preRunTests**:

   - preRunTests is a  default `set of` models runs, to provide useful `sanity checks` and `optimizations` and `details providing runs`. If preRunTests hasen't been executed already, is going to be executed automatically before the `.fit` method.
   - 1. **FastDevRun:** This runs the entire pipeline, `checking` for any `errors` that might cause the `whole process` to fail. This `allows` you to start the training, `walk away`, and come back later `without worrying` about unexpected failures.
     2. **overfitBatches:** This run trains the entire model on a single batch, a popular technique for quick `sanity checks` and shows model's ability to provide good result.
     3. **Profiler:** This run helps identify `potential bottlenecks` in the training pipeline. It provides a `TensorBoard visual chart` showing the time consumption of each step. While it's usually hard for people to make a this chart, but in this project is provided `effortlessly`.
     4. **Learning Rate & Batch Size Optimization:** Automatically finds the optimal learning rate and batch size for your data loader, improving training efficiency.

5. **Simplified experience with smart defaults**:

   - This project not only simplifies the pipeline but also includes `helpful defaults`. 
   - **Useful Info Messages:** Note when these helpful defaults are automatically included, you `receive informative messages`, so you may get reminded to consider either turning them off or customizing their values.

   1. **General regularization** (applied on all layers)
   2. **Best Model save on best `val` loss**
   3. **Learning rate warm up**
   4. **Early Stopping**
   5. **Gradient Clipping**
   6. **Automatically setting GPU devices based on machine**

6. **Regularization:**

   - In `tensorflow` regularization can be applied on each layer, but pytorch doesn't support this. also pytorch doesn't support `L1` reguarization. but these are available in this project.

7. **Special modes:**

   - It provides customized ready pipelines to utilize `dropoutEnsembleMode` and `varational autoencoder mode`, even can used together.

8. **standalone model load:**

   - PyTorch Lightning typically just saves model parameters. This project uniquely saves the **class definitions** in the checkpoint, enabling direct loading **without requiring the original model code**. This provides a standalone, self-contained model.

9. **Device Support**:

   - Leverages `MPS devices` for `Apple M1 processors`, capitalizing on recent advancements in PyTorch to accelerate performance.
   - `Addresses limitations of MPS devices` by incorporating workarounds for unsupported 64 decision data types.

- Additional Extra Info:
  - brazingTorch is designed as a **framework**, following a common approach for building pipelines like this. Similar to PyTorch Lightning, it offers a structured and reusable foundation for building and managing the components involved in the pipeline.
  - note this project only utilizes pytorch lightning, just for the case the users wants to use `multiple gpus` or `distributed computations` provided by pytorch lightning, otherwise all pipeline could have designed from scratch. so finally noting that `multiple gpus` or `distributed computations` are also supported.
  - This project utilizes PyTorch Lightning primarily for its `multi-gpu` and `distributed computing` capabilities, otherwise, the entire pipeline could have been built from scratch. so it can be understood brazingTorch supports this.




---

### models<a id='modelsCurrentlySupported'></a>

models currently supported:

- **Temporal Fusion Transformers**
- **n-hits**
- **n-beats**
- **deepAr**
- **Multivariate Transformers**
- **Univariate Transformers**

could be found at `\models`

### dataPrep & commonDatasets<a id='dataPrepNCommonDatasetsFeatures'></a>

- **Processing Toolkit:** A collection of functions named `dataPrep` is available to help `prepare time series data` for modeling purposes. These functions are located in the `\dataPrep` directory.
- **Processed data to benchmark:** `\commonDatasets\commonDatasetsPrep` directory contains functions that leverage the `dataPrep` functionalities. These functions provide `ready preprocessed` data of some famous time series datasets, can be `used as` data to `benchmark` the performances of your models. Note their `dataset` and `dataloader` `objects` also exist there.

#### features

- **Powerful Tools for Time Series Data Preparation**

  - There are lots of useful functions in `\dataPrep\preprocessing.py` which help to `transform` `raw data` and `do data preparation operations`.

- **`VAnnTsDataset` specialized time series dataset:**

  - Utilizes `parallel` capabilities of pytorch dataset

  - **Efficient `sequence generation` with `getBackForeCastData`:**

    The `getBackForeCastData` function offers a convenient way to generate different types of sequences from `pandas DataFrames`:

    - **Cast modes:** The function supports various modes for generating sequences, including `backcast`, `forecast`, `fullcast`, and `singlePoint`.
    - **Faster than pandas:** Compared to using pandas DataFrames directly, `getBackForeCastData` is specifically designed to utilize a `wrapper around pandas dataframe` but making it `faster` to fetch data.
    - **Mitigates Errors:**  So many errors that can occur during the sequence generation process but it helps mitigate those errors, resulting to save time and effort.

  - **Efficient Data Handling with nested dictionaries:**

    - **Allows Easy Data Structuring with Nested dictionaries:** While `using dictionaries` is an intuitive way to `structure` your `data`, especially `nested dictionaries`(**dictionaries within dictionaries**) `for more complex data`. but `nesting` dictionaries make it `harder` `to do operations`, therefore helps with doing operation on nested dictionaries.
    - **Automatic Handling of Nested Dictionaries:** Forget manual manipulation! This dataset `automatically detects` whether your data is in dictionaries or nested dictionaries. It handles these structures for `loss calculations` and even `converts them to tensors` for your model.
    - **GPU Memory Efficient Tensor Conversion:** This dataset `prioritizes memory efficiency`. It only converts your data to tensors at the `final step`, right before feeding it to the model. This `avoids unnecessary GPU memory usage` and optimizes allocation.

- **Time Series Splitting:**

  - Offers functions to split time series data into training, validation, and testing sets, `even shuffled`.
    - Can `split into three sets` (train, validation, and test) instead of the usual two, making the splitting much easier. Note the most important thing here is that it `considers the sequential nature of time series data`, crucial for accurate preparation.
    - Includes a dedicated function for splitting `"NSeries"` timeseries data, where `multiple time series` reside `within a single dataframe` along each other in different rows.
    - This is an `unmatched unique feature`.

- **Various specialized normalizers:**

  - This project utilizes three main types of normalizers, each with a `subtype` for handling `numerical` and `label data`.

  - also can easily `fit`, `transform` and `inverseTransform` the `all columns` or `single column` of the dataframe.

    1. **SingleColNormalizer:**
       - This is the standard normalizer for individual columns.
       
    2. **MultiColNormalizer:**
       - This normalizer treats `multiple columns` `as one` during normalization.
       
       - It `calculates statistics` (mean, classes, etc.) `across all` specified columns for normalization.
       
    3. **MainGroupSingleColNormalizer:**
    
       - This specialized normalizer is designed for `"NSeries"` data.
       - NSeries refers to data containing multiple distinct time series within a single column.
       - The normalizer `assigns` and manages a separate `SingleColNormalizer` for `each unique group` within the NSeries data. This ensures proper normalization for each individual time series.

- **Supports Diverse Exogenous Variables:** 

  it can work with exogenous variables and variety of them, `static(time unvarying)/time varying`|`known/unknown in future`.

  

- additional info:

  - `dataprep` is designed as a `library` that `requires` you to `manually apply` `suited functions` from the variety of its functions. While some other packages offer `automated data preprocessing`, `researchers` often need to `control` feature selection and `specify` what `operations` to be applied on each feature. Automating this step can create a `"black box"` effect, where you `lose understanding` of the data preparation process. This `lack of transparency` can be `detrimental` for AI researchers who need to have a deep grasp of the preprocessing steps.


# Machine Learning<a id='machineLearning'></a>

**versatileAnn** aims to provide an `easy-to-use` tool for testing `various machine learning models` with different `hyperparameters`.

this section aims for `forecasting` machine learning application

- features:

  - comprehensive gird search on machine learning classic models.

  - apply `cross fold validation` 

  - calculate both 'train' and 'test' scores for multiple metrics separately(most packages don't provide such)

  - parallelized for faster computations

  - memory efficient but can be improved a bit

- additional infos:
  - this section is adapted from https://github.com/FarhangAmaji/binary-classification-Telco-Churn
  - 

---

# Important Message

**The project has not officially been launched:** Many sections of the project have been completed, but not all sections, especially the `\examples` and `\models` sections, need to be updated with the latest reconstruction of `brazingTorch`.

- note the past code exist but not guaranteed to work perfectly.

### Project additional infos<a id='additionalInfos'></a>

- As the name `"VersatileAnn"` suggests, the project was intended to be beneficial across different deep learning **subfields**, including natural language processing (**NLP**), **computer vision**, **forecasting**, and other more **general use cases.** But `models`, `dataPrep` and `commonDatasets` sections are currently only `specialized` for `forecasting` and some NLP use cases.
- `sections` are designed to `work together` seamlessly, leveraging and `support` `each other's` `functionalities`.
- For a deeper dive, `each section` within the project has its own dedicated **readme.md** file explaining these features in detail.  
- **`models`, `dataPrep` and `commonDatasets` sections currently support only Forecasting and NLP but they have Potential for Broader Application:**

  While the current implementation `focuses primarily` on `forecasting` and `NLP`, the project `exhibits` the `potential` to `extend its capabilities` to `other subfields`. `Novel features` within the project `lay` the `groundwork for future` integration of functionalities `tailored` for data preparation in `NLP` and `computer vision`, as well as the potential for developing `models` specific to `computer vision` tasks.
- **clean code & clean structure:**
  - This project prioritizes `code clarity & readability` and maintainability by adhering to best practices for `clean code`. 
  - **Public and Private Functions Naming Convention:** A clear `naming convention` is implemented for `public and private functions`, methods, and classes. `Private elements`, names start with '`_`', are `not intended` for `direct user use`. 
    - **Better know what Functions to use:** this naming convention allows you to readily `identify` the `primary functionalities` and guiding you towards the appropriate functions, methods, or classes to get started. Also simplifies project navigation.

  - Also on different comments it's been indicated that in more comprehensive explanations exist in `\devDocs\codeClarifier` files.

- **conventions:**
  - This project utilizes comments extensively to enhance code readability and understanding. 
  - **cccUsage:** Comments tagged with `cccUsage` offer specific guidance on how to `leverage` the project's `functionalities`. 
  - **cccc Comments:** comments numbered `ccc1` to `ccc4` indicate their relative `importance` or the `scope they influence`, aiding you to `focus` on `more important comments`.
  - **Colored Comments:** To further enrich your coding experience, consider incorporating colored comments into your development environment. The project provides references for these color-coding conventions:
    - **PyCharm**: `\devDocs\conventions\convention_todoTypes.py`

    - **VSCode** better comments extension *settings.json*: `\devDocs\conventions\conventions.md`

  - **set project settings(pycharm):** to have same settings `specially` the `formatter` used in development of this project, in `pycharm` import `\devDocs\versatileAnn_pycharmFormatter.editorconfig`.

- Reading tests in `\tests` folder can be useful in order to understand the project more.

---

# How to get started<a id='HowToGetStarted'></a>

- **in order to take a `super quick glimpse` at files:**
  - take a look at files:

    - for models in `\models`
    - `\brazingTorchFolder`
    - `\commonDatasets\commonDatasetsPrep` which show how data can be processed with `dataPrep`
    - `\dataPrep` also `\dataPrep\normalizers`
    - `\projectUtils\dataTypeUtils` and `projectUtils\misc.py`

---
#### setup

- Definitely the **first thing to do** is to **clone** the project with `git clone https://github.com/FarhangAmaji/versatileAnn`

- run it in docker

  - windows Run docker-desktop, other system skip this step.
    in project folder in cli:
    docker build -t ann .

  * windows cli:
  `docker run -it --rm -v %cd%:/app ann`


  * other systems:
    * `docker run --rm -it -v ${PWD}:/app ann`
      `docker run --rm -it -v $(pwd):/app ann`

- if you are not using docker:

  - then install requirements with `pip install -r requirements.txt`

---

### use cases dependent look at files

in order to know to get started, you should know what do yo want to do and what uses cases of this project do you want to use?!

1. **use for timeseries data preparation:**<a id='useForTsDataPrep'></a>

   - note depending on how you understand and learn new things:
     1. you may go from 1 to 5 in list below.
     2. or you may only start at 5 (commonDatasetsPrep) which utilizes the first four.

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

2. **use BrazingTorch:**<a id='useBrazingTorch'></a>

   - it's a better alternative for pytorch and pytorch lightning and you may need:
     1. take a look at arguments at `\brazingTorchFolder\brazingTorch.py`
        - note there it's been addressed where implementation of each arguments is.
        - note for better `separation of concerns`, most of their implementations are at files in `\brazingTorchFolder\brazingTorchParents`. so reading these files would help understand how exactly this package works.
        - also there are some custom layers at `\brazingTorchFolder\layers\customLayers.py`
        - and custom callbacks at `\brazingTorchFolder\callbacks.py`

3. **models:**<a id='models'></a>

   - If you want get familiar to some latest `state of the art` models in forecasting you may take a look at `\models`. it contains the code of those models with compatibility with BrazingTorch.

4. **examples:**<a id='examples'></a>

   - there are examples on how to define models which use prepared data of `\commonDatassets\commonDatasetsPrep` or some examples have random data. then either use the models in `\models` or create a `custom model`.

5. **machineLearning:**<a id='machineLearning'></a>

6. **projectUtils:**<a id='projectUtils'></a>

   - complementary functions are at `\projectUtils`
   - you may look at files at `\projectUtils\dataTypeUtils` and also `projectUtils/misc.py`

---

# Appendix

#### what is forecasting<a id='whatIsForecasting'></a>

Forecasting leverages `statistical`, `machine learning`, and `deep learning` techniques to analyze past **quantitative** data and `predict future values`. You may be familiar with regression; `regression` is a `famous classic forecasting model`.

- Time series forecasting is ubiquitous in various domains, such as retail, finance, manufacturing, healthcare and natural sciences, with roughly detailed examples below, you may get a better picture.

  - **Engineering**: 

    `Predicting` forces and movement (`forces`, `momentum`, `displacement`, and `velocity`) for designing structures and machines.

  - **Supply Chain Management & Demand Forecasting**: 

    `Estimating` `store sales` or `demands` for `optimized inventory management` can involve predicting both electricity demand and precipitation for efficient planning and `resource allocation`.

  - **Market Analysis:**

    While valuable for informed decision-making, it's crucial to remember the inherent uncertainty of the future, especially in areas like **stock market predictions**.
  
  [go back to "Who this project is for?!"](#whoThisProjectIsFor)

---

readme version: 1.0.0

- kkk this will updated after official launch

---

- note currently `\models` don't work as they re not yet changed to work with newer version of brazingTorch

- examples are not workings, need to be updated

- machineLearning has not been committed

- kkk this part should be cleaned

- 

  ---

- kkk there is no difference between **bold** and `highlights`, so there is no logic going on

- kkk GPU Memory Efficient Tensor Conversion should be benchmarked(wait wait does benchmark may any sense, it just need to be tested)

- kkk complete machine learning section features

- kkk revise machine learning section features

- kkk add machine learning to How to get started

- kkk add picture from predictions

- kkk add explanation to this readMe about examples

- kkk fill brazingTorch repo with this readMe

- kkk add it has models with exogenous variables and explainability

- kkk grammar check for whole file

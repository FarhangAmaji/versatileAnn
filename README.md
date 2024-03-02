# what versatileAnn is

While developing machine learning models can be time-consuming, even the smallest typos in programming can derail months of hard work, rendering unreliable results. Utilizing `well-tested tools` can significantly improve the experience of both using and developing models.

This project aims to provide utility tools that streamline the development of models, particularly those in the realm of deep learning. As the name `"VersatileAnn"` suggests, the project strives to be beneficial across different deep learning subfields, including natural language processing (NLP), computer vision, forecasting, and other more general use cases. Not only can it be useful for people to `simply try out some models`, but it's designated to be especially handy for `AI researchers` who require a robust and efficient development environment.

The project comprises four distinct sections: `dataPrep`, `brazingTorch`, `models`, and `machineLearning`. `BrazingTorch` and `machineLearning` offer valuable resources for individuals aspiring to delve into development and utilizing deep learning and machine learning approaches, respectively.

However, the dataPrep and models sections are currently `specialized` for `forecasting` containing many of its `state of the art` models. It's worth noting that NLP falls under the category of models dealing with sequential data, where information is processed in a specific order. This alignment allows VersatileAnn to be particularly useful for NLP tasks. The project holds potential for incorporating novel ideas similar to the existing ones, but these are yet to be implemented for other subfields.

Additionally, VersatileAnn aims to provide an easy-to-use tool for testing various machine learning models with different hyperparameters. This functionality can significantly save development time and effort by streamlining the process of identifying the optimal configuration for a specific model.

## sections

Note each section has its own readme.md.

### brazingTorch

`brazingTorch` was developed here in this project, but as it offers valuable functionalities for the broader deep learning community, so it was moved to its own github repo: https://github.com/FarhangAmaji/brazingTorch

it can be found at `/brazingTorchFolder` .

Cool features:

- Note many features mentioned below have their options and may be turned off.

1. **Streamlined Deep Learning Pipeline:**

   - BrazingTorch `simplifies` model development by providing a `customizable pipeline`.
   - In `most cases`, you only need to define the forward pass and call the `fit` method, making it remarkably user-friendly.
   - For further customization, the `commonStep` method can be overwritten to suit specific needs.

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

   1. general regularization(applied on all layers)
   2. best Model save on best `val` loss
   3. learning rate warm up
   4. early stopping
   5. gradient clipping

6. **Regularization:**

   - In `tensorflow` regularization can be applied on each layer, but pytorch doesn't support this. also pytorch doesn't support `L1` reguarization. but these are available in this project.

7. **Special modes:**

   - It provides customized pipelines to utilize `dropoutEnsembleMode` and `varational autoencoder mode`, even can used together.

8. **standalone model load:**

   - PyTorch Lightning typically just saves model parameters. This project uniquely saves the **class definitions** in the checkpoint, enabling direct loading **without requiring the original model code**. This provides a standalone, self-contained model.

9. **Easier optimizer/scheduler setup:**

   - Dedicated properties simplify configuration compared to PyTorch Lightning.

10. **Pipeline helper functions**: Offers some methods to work easier in framework format

11. **Color Coding for Clarity: Warnings, Errors, and Infos**

brazingTorch is a high level wrapper around `pytorch lightning`, which itself is a wrapper around `pytorch`. therefore, you can seamlessly integrate BrazingTorch with other PyTorch, PyTorch Lightning, and even TensorBoard features for a tailored deep learning experience.

note this project only utilizes pytorch lightning, just for the case the users wants to use `multiple gpus` or `distributed computations` provided by pytorch lightning, otherwise all pipeline could have designed from scratch. so finally noting that `multiple gpus` or `distributed computations` are also supported.

This project utilizes PyTorch Lightning primarily for its `multi-gpu` and `distributed computing` capabilities, otherwise, the entire pipeline could have been built from scratch. so it can be understood brazingTorch supports this.

brazingTorch is designed as a **framework**, following a common approach for building pipelines like this. Similar to PyTorch Lightning, it offers a structured and reusable foundation for building and managing the components involved in the pipeline.

---

kkk brazingTorch is a framework but dataprep libray making it not a blackbox ; as AI researcher really need to know what exactly is going on on the data preprocessing steps, or model

kkk good clean code practices focusing on readability. kkk public and private mthod/class naming

kkk cccUsage

kkk good detailed comments with numerating them by importance; add colors

kkk this is a big project more than 15,000 lines of code. so please be patient to get familiar with this project. but it can be guessed with cool features and features to prevent errors, it really worth it to invest some time to it.






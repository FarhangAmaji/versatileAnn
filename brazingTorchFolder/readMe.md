this readMe `supposes` that you `have read` the `README.md` in the root folder.

in the file `brazingTorch.py` you can find the `BrazingTorch` class, needed to create `pipeline` and `run` your `models easily`. You should inherit from it, and define `forward` method.

for further customization for your pipeline you can redefine `commonStep`, note it's been recommended to get ideas from its original implementation.

kkk how to find functionalities this project provide:

```tex
how to find functionalities this project provide:
		"take a look at files in `\brazingTorchParents`"
```

kkk add order of files based on usual usage in `\brazingTorchParents`

kkk read more about options

note to have cleaner architecture and separation of concerns brazingTorch has been splitted to some parent (base) classes (existing in `\brazingTorchParents`) each one taking one responsibility. 

- also note each file there has related public methods which you can use with this project. so it's recommended to take a look at files in `\brazingTorchParents` and read their important comments (usually tagged with `cccUsage`)
- note `ccc1` or `ccc2` also add most important about how the code is implemented, so in the case that implementation is important for you or you want to customize the code further, you may want to have these comments read.

- the private internal methods used in brazingTorch exist in `\brazingTorchParents\innerClassesWithoutPublicMethods`

for cleaner architecture you may

- has very detailed names

- read `cccUsage` comments
- kkk indepent preRunTests

you 

use cases:
## comments convention

---

- the numbering in the comments goes from `1` the most important to `4` the ordinary; this is up to the developer judgement

  - note probably the `scope` which the code `affects` is also a factor in determining the importance of comment

- ----

  #mustHave explanations for `mustHave`s, `goodToHave`s, `bug`s and `bugPotn`s

- general tag to check later:

  - `kkk`
  - color: `FF0000`

- explanation comments:

  - `ccc`

    - cases:

      1. the comments in general `should not` explain `what` a line is doing, `unless`(except in the cases):

         - note it should be preferred `not to have comment` for `what is doing` `unless` the comment `really helps`
         - note the `blocks` are primarily preferred to have a `function(with a detailed named)` for them

         1. the line syntax is not commonly known, for i.e. it's a rare syntax which is not widely used
         2. can explain `few simple lines` in just `one look`
         3. complex lines

      2. expalin why a block is doing sth:

         - the code preferably should have tagLines like # Lmwf, so the user can search that in the whole project(ctrl+shift+F) and find where that line is
         - note for more complex explanations, it's better to create a codeClarifier file in devDocs\codeClarifier
         - explain why, 1. some details on the other parts 2. what feature was intended to be designed: have pushed the developer to do the code this way

    - colors:

      - `ccc1` `FFFF00`
      - `ccc2` `FFFF2A`
      - `ccc3` `FFFF54`
      - `ccc4` `FFFF7E`

  - `cccUsage`

    - to explain for `users` to how use the feature of the code
    - cases:
      1. what type of args should be passed to funcs
    - color: `FF00FF`

  - `cccDev`

    - usually explains high level overview on why the code is intended to be structured this way, and why other approaches are not utilized
    - note this is intended to be used rarely and the most of the algorithm or programming related explanations would fit into `ccc`
      - color: `FFFF54`

- features to be added

  - mustHave
    - mustHave1 00FF00
    - mustHave2 28FF28
    - mustHave3 50FF50
  - goodToHave
    - goodToHave1 2828FF
    - goodToHave2 5050FF
    - goodToHave3 7878FF
    - restructureNeeded 5050FF

- bugs

  - bugs known:
    - `bug 1~3`
    - colors:
      - `bug1` `FF0000`
      - `bug2` `FF2A00`
      - `bug3` `FF5400`
  - bugs potential:
    - cases:
      - during the development we are not sure does a line works or not
      - is this section compatible with the other parts of the code
    - `bugPotn 1~3` also `bugPotn_hardcoded`
    - colors:
      - `bugPotn1`  `FF0064`
      - `bugPotn2`  `FF2A64`
      - `bugPotn3`  `FF5464`
      - `bugPotn_hardcoded` `FF2A64`

- places which need tests

  - `addTest1` `FF0000`
  - `addTest2` `FF2A00`
  - `addTest3` `FF5400`

- qqq

  - in development there may be some doubts, for example on:

    - this should be used for `temporary` purposes and should `not exist` in `final`

    1.  does this line of code works or not(this is most likely is going to occur when the is not following `test driven development`)
        - note if the doubt probably is not temporarily, the `bugPotn` should be used instead
    2.  not sure about `compatibility` with the other parts of the code

  - color `FF2A64`

- style `9696FF`

---

# commits prefix conventions

- note again numbering goes from `1` the most important to `4` the ordinary

- ---

  `feat` 1~4 : new feature is added

- `improve` 1~4 : existing feature is improved

- `change` 1~4 : existing feature is changed

- `restruct`: the big part of changed, and which makes other parts also to be changed

- `refactor` 1~4 : refactoring

  - examples:
    - changing the orders of arguments in a function
    - renaming some variable
    - putting some block of code to a function

- `test`

- `docs`

- `style`

- `chore`

## set comments/todos in vscode

in `vscode` the comments use `better comments` extension with these added style to `settings.json`

```
"better-comments.tags": [
    {
      "name": "kkk",
      "description": "Follow-up required",
      "foreground": "#FF0000"
    },
    {
      "name": "ccc1",
      "description": "most important comments probably with bigger scope",
      "foreground": "#FFFF00"
    },
    {
      "name": "ccc2",
      "description": "important comments",
      "foreground": "#FFFF2A"
    },
    {
      "name": "ccc3",
      "description": "normal comments",
      "foreground": "#FFFF54"
    },
    {
      "name": "ccc4",
      "description": "trivial comments",
      "foreground": "#FFFF7E"
    },
    {
      "name": "cccUsage",
      "description": "how to use for users",
      "foreground": "#FF00FF"
    },
    {
      "name": "cccDev",
      "description": "high level overview explanation for Developers",
      "foreground": "#FFFF54"
    },
    {
      "name": "mustHave1",
      "description": "Must Have important",
      "foreground": "#00FF00"
    },
    {
      "name": "mustHave2",
      "description": "Must Have quite important",
      "foreground": "#28FF28"
    },
    {
      "name": "mustHave3",
      "description": "Must Have but not as first priority",
      "foreground": "#50FF50"
    },
    {
      "name": "goodToHave1",
      "description": "Good to Have a cool feature",
      "foreground": "#2828FF"
    },
    {
      "name": "goodToHave2",
      "description": "Good to Have a feature which can be useful",
      "foreground": "#5050FF"
    },
    {
      "name": "goodToHave3",
      "description": "Good to Have for nicer interface",
      "foreground": "#7878FF"
    },
    {
      "name": "restructureNeeded",
      "description": "Good to restructure the code",
      "foreground": "#5050FF"
    },
    {
      "name": "bug1",
      "description": "Bug known critical/urgent",
      "foreground": "#FF0000"
    },
    {
      "name": "bug2",
      "description": "Bug known",
      "foreground": "#FF2A00"
    },
    {
      "name": "bug3",
      "description": "Bug known but doesn't make problem which fails",
      "foreground": "#FF5400"
    },
    {
      "name": "bugPotn1",
      "description": "Potential Bug 1",
      "foreground": "#FF0064"
    },
    {
      "name": "bugPotn2",
      "description": "Potential Bug 2",
      "foreground": "#FF2A64"
    },
    {
      "name": "bugPotn3",
      "description": "Potential Bug 3",
      "foreground": "#FF5464"
    },
    {
      "name": "bugPotn_hardcoded",
      "description": "bugPotn where sth is hardcoded",
      "foreground": "#FF2A64"
    },
    {
      "name": "addTest1",
      "description": "Additional Test 1",
      "foreground": "#FF0000"
    },
    {
      "name": "addTest2",
      "description": "Additional Test 2",
      "foreground": "#FF2A00"
    },
    {
      "name": "addTest3",
      "description": "Additional Test 3",
      "foreground": "#FF5400"
    },
    {
      "name": "qqq",
      "description": "temporary doubt questions",
      "foreground": "#2A64FF"
    },
    {
      "name": "style",
      "description": "styling comments",
      "foreground": "#9696FF"
    }
  ],
```

## naming convention

- safety:
  - naming `func` on what `exactly` they `are doing`
  - name private funcs starting with `_` or `__`(its important we define what funcs are meant to be used by user and what funcs are intended to be used internally)
  - add units like miliseconds to var or func names
- cleanness:
  1. name with `_` to make it more `eyecatching` about important details(`take a look at how the names are chosen` currently in the project)
  2. names should describe `intent`, `why its doing` and not `what is doing`; `self explanatory`
     - names should describe:
       1. `why exists`
       2. `what it does`
       3. `how its used`
       4. context
  3. make names as `short as possible`, `long as needed`
  4. name constants
  5. name conditions
  6. dont use vague names like `controller` or `layer`
  7. verb for func names
  8. dont use abbreviations; some times context tricks u this abbreviation is ok
  9. `PascalCase` for `classes

## formatting in vscode

#mustHave1 the formatting config of `pycharm` must be added to conventions

the formatting is done with a formatter in `vscode` with following `settings.json`

```
"[python]": {
  "python.formatting.autopep8Args": [
    "--max-line-length=100",
    "--indent-size=4",
    "--indent-after-paren=1",
    "--aggressive",
    "--max-args=3"
  ]
}
```

## formatting convention

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

## comment convention

---

todos

- not corrected todo tag
  - kkk FF0000
- bugs
  - bugCriticalUnknown FF0000
  - bugKnown1 FF0000
  - bugKnown2 FF8C00
  - bugPotentialCheck1 FF5000
  - bugPotentialCheck2 FF8C00
  - addTest1 FF0000
  - addTest2 FF8C00
- features to be added
  - mustHave
    - mustHave1 00FF00
    - mustHave2 32FF32
    - mustHave3 64FF64
  - goodToHave
    - goodToHave1 3232FF
    - goodToHave2 6464FF
    - goodToHave3 9696FF
  - styles
    - style1 00FF00
    - style2 9696FF
- questions
  - qqq
- comments
  - devComments
    - cccDevStruct FFFF00
    - cccDevAlgo FFFF00
  - userComments
    - cccAlgo FF00FF
    - cccUsage FF00FF

---

the devs should not be included in `master branch`

---

- cccDevStruct:

  explains coding parts, for i.e. if some abstract class is needed or how objects should be, etc

- normal comments:

  - explains what the code is doing for few next lines, or less important (depending of scope of impact of lines) why its done this way

- cccAlgo:

  - explains what the code is doing for users. and the scope of importance is bigger

- cccDevAlgo :

  - similar to `normal comments` but with more importance or scope

- cccUsage:

  - usually is about what type of args should be passed to funcs

---

the comments use `better comments` extention with these added style to `settings.json`

```
"better-comments.styles": [
    /*    # bugs */
        {
            "text": "bugCriticalUnknown",
            "color": "#FF0000"
        },
        {
            "text": "bugKnown1",
            "color": "#FF0000"
        },
        {
            "text": "bugKnown2",
            "color": "#FF8C00"
        },
        {
            "text": "bugPotentialCheck1",
            "color": "#FF5000"
        },
        {
            "text": "bugPotentialCheck2",
            "color": "#FFDC00"
        },
        {
            "text": "addTest1",
            "color": "#FF0000"
        },
        ,
        {
            "text": "addTest2",
            "color": "#FF8C00"
        },
    /*    # features to be added */
    /*    # - mustHave */
        {
            "text": "mustHave1",
            "color": "#00FF00"
        },
        {
            "text": "mustHave2",
            "color": "#32FF32"
        },
        {
            "text": "mustHave3",
            "color": "#64FF64"
        },
    /*    # - goodToHave */
        {
            "text": "goodToHave1",
            "color": "#3232FF"
        },
        {
            "text": "goodToHave2",
            "color": "#6464FF"
        },
        {
            "text": "goodToHave3",
            "color": "#9696FF"
        },
    /*    # - styles */
        {
            "text": "style1",
            "color": "#00FF00"
        },
        {
            "text": "style2",
            "color": "#9696FF"
        },
    /*    # comments */
    /*	# - devComments */
        {
            "text": "cccDevStruct",
            "color": "#FFFF00"
        },
        {
            "text": "cccDevAlgo",
            "color": "#FFFF00"
        },
    /*	# - userComments */
        {
            "text": "cccAlgo",
            "color": "#FF00FF"
        },
        {
            "text": "cccUsage",
            "color": "#FF00FF"
        },
    /*	# questions */
        {
            "text": "qqq",
            "color": "#9696FF"
        }
    ],
```

## formatting convention

- safety:
  - naming `func` on what `exactly` they `is doing`
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

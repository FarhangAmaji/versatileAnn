## formatting convention

the formatting is done with a formatter in `vscode` with following `settings.json`

```
#kkk this one should be added
```

## comment convention

#kkk I should define `todo types` depending on `priorites`(like importanlevel1 or style or comment )

the comments use `better comments` extention with these added style to `settings.json`

```
###### rn are not defined
"better-comments.styles": [
    {
        "text": "CustomStyle1",
        "color": "#FF5733",
        "icon": "rocket"
    },
    {
        "text": "CustomStyle2",
        "color": "#33FF57",
        "icon": "flame"
    }
]
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

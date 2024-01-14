"""
in the past this module had some funcs used to find all parent classes of a class
but they didn't fulfill the purpose to be general enoguh. so I removed them.
"""
# goodToHave2
#  later make this module general enough to be used in other projects

def _findParentClasses_OfAClass_tillAnotherClass(cls_, classTillThatIsWanted,
                                                 parentClasses: dict = None):
    # this is similar to
    parentClasses = parentClasses or {}
    # goodToHave3 bugPotentialCheck2
    #  some classes may have same .__name__ but are actually different classes
    #  but I am not counting for that(maybe later).
    #  so for now each class is going to be captured in a dict with {class.__name__:classObj}

    if str(cls_) == str(classTillThatIsWanted):
        return parentClasses
    elif cls_ is object:
        return parentClasses

    parentClasses.update({cls_.__name__: cls_})
    parentsOfThisClass = cls_.__bases__
    for potc in parentsOfThisClass:
        parentClasses = _findParentClasses_OfAClass_tillAnotherClass(
            potc,
            classTillThatIsWanted,
            parentClasses)
    return parentClasses

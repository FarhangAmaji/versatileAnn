def areItemsOfList1_InList2(list1, list2, giveNotInvolvedItems=False):
    notInvolvedItems = []

    setList2 = set(list2)
    for item in list1:
        if item not in setList2:
            notInvolvedItems.append(item)

    result = notInvolvedItems == []
    if giveNotInvolvedItems:
        return result, notInvolvedItems
    return result


def isListTupleOrSet(obj):
    return isinstance(obj, (list, tuple, set))


def isIterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def listToRanges(inputList):
    if not inputList:
        return []

    ranges = []
    start = inputList[0]
    end = inputList[0]

    for num in inputList[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(range(start, start + 1))
            else:
                ranges.append(range(start, end + 1))
            start = end = num

    if start == end:
        ranges.append(range(start, start + 1))
    else:
        ranges.append(range(start, end + 1))

    return ranges


def listRangesToList(rangeList):
    if not rangeList:
        return []

    if not all(isinstance(rg, range) for rg in rangeList):
        raise ValueError('Not all items are ranges')

    res = []
    for rg in rangeList:
        res.extend(range(rg.start, rg.stop))

    return res


def hasThisListAnyRange(list_):
    return any([type(item) == range for item in list_])


def similarItemsString(inputList):
    result = []
    lastItem = inputList[0]
    count = 1

    def formatItem(item, count):
        itemStr = f"'{item}'" if isinstance(item, str) else str(item)
        string = f"{count} * [{itemStr}]" if count > 1 else itemStr
        return string

    for item in inputList[1:]:
        if item == lastItem:
            count += 1
        else:
            result.append(formatItem(lastItem, count))
            lastItem = item
            count = 1

    result.append(formatItem(lastItem, count))

    result2 = []
    currRes2 = []
    currResFormat = lambda currRes2: '[' + ', '.join(currRes2) + ']'

    for item2 in result:
        if '*' not in item2:
            currRes2.append(item2)
        else:
            if currRes2:
                result2.append(currResFormat(currRes2))
                currRes2 = []
            result2.append(item2)
    if currRes2:
        result2.append(currResFormat(currRes2))

    return ' + '.join(result2)

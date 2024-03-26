def generate_tuple(start, target, gap_char):
    merged = []
    for lt1, lt2 in zip(start[1:], target[:-1]):
        if lt1 != gap_char:
            merged.append(lt1)
        elif lt2 != gap_char:
            merged.append(lt2)
        else:
            merged.append(lt1)

    # Append the first element of start and the last element of target
    return [start[0]] + merged + [target[-1]]


def match_strings(string_list, pattern, gap_char):
    count = 0
    for edge in string_list:
        string = edge[0]
        weight = edge[1]
        if len(string) != len(pattern):
            continue
        match = True
        for i in range(len(string)):
            if pattern[i] != gap_char and pattern[i] != string[i]:
                match = False
                break
        if match:
            count += weight
    return count
